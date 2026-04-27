#!/usr/bin/env python3
"""
Step 7: Predict Docking from Two Unbound PDB Files
===================================================
Usage:
    # Predict only (saves predicted complex PDB)
    python step7_predict.py receptor.pdb ligand.pdb

    # Predict and evaluate against ground truth
    python step7_predict.py receptor.pdb ligand.pdb --bound bound_complex.pdb

    # Specify chain IDs manually
    python step7_predict.py receptor.pdb ligand.pdb --rec_chains A B --lig_chains C

    # Use a specific model checkpoint
    python step7_predict.py receptor.pdb ligand.pdb --checkpoint path/to/best_model.pt

    # Specify output path
    python step7_predict.py receptor.pdb ligand.pdb -o predicted_complex.pdb
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Superimposer
from scipy.spatial.distance import cdist

from step3_feature_extraction import ProteinFeatureExtractor, AA_TO_INDEX, AA_PROPERTIES
from step4_model import ProteinDockingModel
from step6_evaluate import compute_fnat, compute_lrmsd, compute_irmsd, compute_dockq, capri_quality


def extract_features_from_pdb(pdb_path, chain_filter=None):
    """
    Extract 26D node features and contact graph from a PDB file.

    Args:
        pdb_path: path to PDB file
        chain_filter: list of chain IDs to use (None = all)

    Returns:
        dict with coords, node_features, edges, edge_distances, num_residues
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = next(structure.get_models())

    coords = []
    one_hot = []
    properties = []

    for chain in model:
        if chain_filter is not None and chain.id not in chain_filter:
            continue
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            if 'CA' not in residue:
                continue
            coords.append(residue['CA'].get_coord())

            res_name = residue.resname
            oh = [0.0] * 20
            if res_name in AA_TO_INDEX:
                oh[AA_TO_INDEX[res_name]] = 1.0
            one_hot.append(oh)
            properties.append(AA_PROPERTIES.get(res_name, [0, 0, 0]))

    coords = np.array(coords, dtype=np.float32)
    one_hot = np.array(one_hot, dtype=np.float32)
    properties = np.array(properties, dtype=np.float32)
    node_features = np.concatenate([coords, one_hot, properties], axis=1)

    # Contact graph (8 A threshold)
    dist_matrix = cdist(coords, coords)
    i_idx, j_idx = np.where((dist_matrix < 8.0) & (dist_matrix > 0))
    edges = np.stack([i_idx, j_idx], axis=0)
    edge_distances = dist_matrix[i_idx, j_idx]

    return {
        'coords': coords,
        'node_features': node_features,
        'edges': edges,
        'edge_distances': edge_distances,
        'num_residues': len(coords),
    }


def extract_bound_ligand_coords(pdb_path, lig_chains):
    """Extract ligand Cα coordinates from a bound complex PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('bound', pdb_path)
    model = next(structure.get_models())

    coords = []
    for chain in model:
        if chain.id in lig_chains:
            for residue in chain:
                if residue.id[0] != ' ' or 'CA' not in residue:
                    continue
                coords.append(residue['CA'].get_coord())
    return np.array(coords, dtype=np.float32)


def get_chain_ids(pdb_path):
    """Get all chain IDs from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)
    model = next(structure.get_models())
    return [c.id for c in model.get_chains()]


def write_predicted_complex(receptor_pdb, ligand_pdb, pred_lig_coords,
                            rec_chains, lig_chains, output_path):
    """
    Write predicted complex PDB: receptor (unchanged) + transformed ligand.
    Replaces ligand Cα coordinates with predicted positions;
    shifts all other atoms in each residue by the same delta.
    """
    parser = PDBParser(QUIET=True)

    rec_structure = parser.get_structure('rec', receptor_pdb)
    lig_structure = parser.get_structure('lig', ligand_pdb)

    # Build new structure
    new_structure = Structure.Structure('predicted')
    new_model = Model.Model(0)
    new_structure.add(new_model)

    # Add receptor chains unchanged
    rec_model = next(rec_structure.get_models())
    for chain in rec_model:
        if rec_chains is None or chain.id in rec_chains:
            new_model.add(chain.copy())

    # Transform ligand: shift each residue by the delta applied to its Cα
    lig_model = next(lig_structure.get_models())
    ca_idx = 0
    for chain in lig_model:
        if lig_chains is not None and chain.id not in lig_chains:
            continue
        new_chain = chain.copy()
        for residue in new_chain:
            if residue.id[0] != ' ' or 'CA' not in residue:
                continue
            if ca_idx >= len(pred_lig_coords):
                break
            old_ca = residue['CA'].get_vector().get_array()
            new_ca = pred_lig_coords[ca_idx]
            delta = new_ca - old_ca
            for atom in residue:
                atom.set_coord(atom.get_coord() + delta)
            ca_idx += 1
        # Avoid duplicate chain ID
        if new_model.has_id(new_chain.id):
            # Rename chain to avoid collision
            for alt in 'XYZWVU':
                if not new_model.has_id(alt):
                    new_chain.id = alt
                    break
        new_model.add(new_chain)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(output_path))


def predict(receptor_pdb, ligand_pdb, model, device,
            rec_chains=None, lig_chains=None):
    """
    Run docking prediction.

    Returns:
        pred_lig_coords: (M, 3) numpy array of predicted ligand positions
        rec_features: receptor feature dict
        lig_features: ligand feature dict
        confidence: float
    """
    rec_features = extract_features_from_pdb(receptor_pdb, rec_chains)
    lig_features = extract_features_from_pdb(ligand_pdb, lig_chains)

    # Prepare tensors
    rec_data = {
        'node_features': torch.tensor(rec_features['node_features'], dtype=torch.float32).to(device),
        'edge_index': torch.tensor(rec_features['edges'], dtype=torch.long).to(device),
        'coords': torch.tensor(rec_features['coords'], dtype=torch.float32).to(device),
    }
    lig_data = {
        'node_features': torch.tensor(lig_features['node_features'], dtype=torch.float32).to(device),
        'edge_index': torch.tensor(lig_features['edges'], dtype=torch.long).to(device),
        'coords': torch.tensor(lig_features['coords'], dtype=torch.float32).to(device),
    }

    with torch.no_grad():
        rotation, translation, confidence = model(rec_data, lig_data)
        pred_coords = model.apply_transformation(lig_data['coords'], rotation, translation)

    return (pred_coords.cpu().numpy(), rec_features, lig_features,
            confidence.item(), rotation.cpu().numpy(), translation.cpu().numpy())


def find_checkpoint():
    """Auto-detect most recent best_model.pt."""
    script_dir = Path(__file__).parent
    experiments = script_dir.parent / 'experiments'
    candidates = sorted(experiments.glob('*/best_model.pt'))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description='Predict protein-protein docking from unbound structures')
    parser.add_argument('receptor', help='Path to receptor PDB (unbound)')
    parser.add_argument('ligand', help='Path to ligand PDB (unbound)')
    parser.add_argument('--bound', default=None,
                        help='Path to bound complex PDB (for evaluation)')
    parser.add_argument('--rec_chains', nargs='+', default=None,
                        help='Receptor chain IDs (default: all chains in receptor PDB)')
    parser.add_argument('--lig_chains', nargs='+', default=None,
                        help='Ligand chain IDs (default: all chains in ligand PDB)')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PDB path (default: predicted_complex.pdb)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate inputs
    rec_path = Path(args.receptor)
    lig_path = Path(args.ligand)
    if not rec_path.exists():
        print(f"Receptor PDB not found: {rec_path}")
        sys.exit(1)
    if not lig_path.exists():
        print(f"Ligand PDB not found: {lig_path}")
        sys.exit(1)

    # Load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print("No checkpoint found. Use --checkpoint to specify path.")
        sys.exit(1)

    print("=" * 70)
    print("DockGraph — Protein-Protein Docking Prediction")
    print("=" * 70)
    print(f"  Receptor: {rec_path.name}")
    print(f"  Ligand:   {lig_path.name}")
    print(f"  Model:    {ckpt_path}")
    print(f"  Device:   {device}")

    model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Get chain IDs if not specified
    if args.rec_chains is None:
        args.rec_chains = get_chain_ids(rec_path)
    if args.lig_chains is None:
        args.lig_chains = get_chain_ids(lig_path)

    print(f"  Receptor chains: {args.rec_chains}")
    print(f"  Ligand chains:   {args.lig_chains}")

    # Predict
    print("\nPredicting...")
    pred_lig_coords, rec_feat, lig_feat, confidence, rot, trans = predict(
        str(rec_path), str(lig_path), model, device,
        args.rec_chains, args.lig_chains
    )

    print(f"\n  Predicted transformation:")
    print(f"    Rotation (axis-angle): [{rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}]")
    print(f"    Translation (A):       [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")
    print(f"    Confidence:            {confidence:.4f}")

    # Movement summary
    orig_center = lig_feat['coords'].mean(axis=0)
    pred_center = pred_lig_coords.mean(axis=0)
    displacement = np.linalg.norm(pred_center - orig_center)
    print(f"    Ligand displacement:   {displacement:.2f} A")

    # Save predicted complex
    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / 'predicted_complex' / 'predicted_complex.pdb'
    write_predicted_complex(
        str(rec_path), str(lig_path), pred_lig_coords,
        args.rec_chains, args.lig_chains, output_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Predicted complex saved: {output_path}")

    # Evaluate against ground truth if provided
    if args.bound:
        bound_path = Path(args.bound)
        if not bound_path.exists():
            print(f"  Bound PDB not found: {bound_path}")
            return

        print(f"\n{'=' * 70}")
        print("EVALUATION AGAINST GROUND TRUTH")
        print(f"{'=' * 70}")

        native_lig_coords = extract_bound_ligand_coords(
            str(bound_path), args.lig_chains
        )

        n = min(len(pred_lig_coords), len(native_lig_coords))
        if n == 0:
            print("  Could not extract ligand from bound complex.")
            print("  Check that --lig_chains matches chains in the bound PDB.")
            return

        pred_lig = pred_lig_coords[:n]
        native_lig = native_lig_coords[:n]
        pred_rec = rec_feat['coords']
        native_rec = rec_feat['coords']  # receptor is unchanged

        fnat = compute_fnat(pred_rec, pred_lig, native_rec, native_lig)
        lrmsd = compute_lrmsd(pred_rec, pred_lig, native_rec, native_lig)
        irmsd = compute_irmsd(pred_rec, pred_lig, native_rec, native_lig)
        dockq = compute_dockq(irmsd, lrmsd, fnat)
        quality = capri_quality(dockq)

        print(f"\n  CAPRI Metrics:")
        print(f"    DockQ:   {dockq:.4f}  ({quality})")
        print(f"    fnat:    {fnat:.4f}")
        print(f"    I-RMSD:  {irmsd:.4f} A")
        print(f"    L-RMSD:  {lrmsd:.4f} A")

        success = "YES" if dockq >= 0.23 else "NO"
        print(f"\n    CAPRI acceptable or better: {success}")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()