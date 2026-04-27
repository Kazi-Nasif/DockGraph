#!/usr/bin/env python3
"""
predict_casp15.py
=================
Run DockGraph docking refinement on CASP15 targets.

Pipeline:
  1. Load AlphaFold3-predicted monomer structures (from AF3 Server output)
  2. Prepare input complex in DockGraph format (receptor + ligand PDB files)
  3. Run DockGraph inference using the trained model
  4. Output refined complex structure

This script bridges AlphaFold3 predictions to DockGraph's input format.
DockGraph expects:
  - receptor.pdb: Fixed chain (here: CNPase, chain A)
  - ligand.pdb:   Mobile chain (here: Nanobody 7E, chain B)
  
The model predicts a rigid-body transformation (rotation + translation)
to refine the ligand position relative to the receptor.

Usage:
    python predict_casp15.py \
        --af3_output_dir alphafold_outputs/H1141_T206/ \
        --model_path /path/to/DockGraph/best_model.pt \
        --output_dir predictions/H1141_T206/

Requirements:
    - BioPython (pip install biopython)
    - PyTorch with TorchScript support
    - DockGraph model weights (best_model.pt from epoch 42)
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path


def parse_af3_output(af3_dir: str, output_dir: str) -> dict:
    """
    Parse AlphaFold3 Server output and extract individual chain structures.
    
    AlphaFold3 Server outputs:
      - fold_*_model_*.cif : Predicted complex structure(s)
      - fold_*_summary_confidences_*.json : Confidence metrics (pLDDT, pTM, ipTM, PAE)
      - fold_*_full_data_*.json : Detailed per-residue and per-pair data
    
    We need to:
      1. Parse the best-ranked mmCIF complex structure
      2. Split into receptor (chain A) and ligand (chain B)
      3. Convert to PDB format for DockGraph
      
    Parameters
    ----------
    af3_dir : str
        Directory containing AlphaFold3 output files
    output_dir : str
        Directory to save parsed structures
        
    Returns
    -------
    dict
        Paths to receptor.pdb, ligand.pdb, complex.pdb, and confidence metrics
    """
    try:
        from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
    except ImportError:
        sys.exit("ERROR: BioPython required. Install with: pip install biopython")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find mmCIF files (AF3 output format)
    cif_files = sorted(Path(af3_dir).glob("*.cif"))
    if not cif_files:
        # Also check for PDB files
        cif_files = sorted(Path(af3_dir).glob("*.pdb"))
    
    if not cif_files:
        print(f"ERROR: No structure files found in {af3_dir}")
        print(f"Expected AlphaFold3 output files (*.cif or *.pdb)")
        sys.exit(1)
    
    # Parse confidence JSONs to find best model
    json_files = sorted(Path(af3_dir).glob("*confidences*.json"))
    best_model_idx = 0
    best_iptm = 0.0
    
    for jf in json_files:
        with open(jf) as f:
            conf = json.load(f)
        iptm = conf.get("iptm", conf.get("ipTM", 0.0))
        print(f"  Model {jf.name}: ipTM = {iptm:.3f}")
        if iptm > best_iptm:
            best_iptm = iptm
            best_model_idx = json_files.index(jf)
    
    # Use best model (or first if no confidence data)
    structure_file = str(cif_files[min(best_model_idx, len(cif_files) - 1)])
    print(f"  Using structure: {structure_file}")
    
    # Parse structure
    ext = os.path.splitext(structure_file)[1].lower()
    if ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("af3_complex", structure_file)
    model = structure[0]
    
    # Report chains found
    chains = list(model.get_chains())
    print(f"  Chains found: {[c.get_id() for c in chains]}")
    
    class ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id
        def accept_chain(self, chain):
            return chain.get_id() == self.chain_id
    
    io = PDBIO()
    io.set_structure(structure)
    
    result = {"confidence": {"ipTM": best_iptm}}
    
    # Extract chains
    # Convention for DockGraph: 
    #   receptor = larger chain (antigen/CNPase, chain A)
    #   ligand   = smaller chain (nanobody, chain B)
    chain_ids = [c.get_id() for c in chains]
    
    if len(chain_ids) < 2:
        print(f"WARNING: Expected 2 chains, found {len(chain_ids)}")
        print("  AF3 may have predicted chains with different IDs.")
        print("  Attempting to use first two chains as receptor/ligand...")
    
    # Map chains by size (receptor = larger, ligand = smaller)
    chain_sizes = {}
    for c in chains:
        n_res = sum(1 for r in c.get_residues() if r.get_id()[0] == ' ')
        chain_sizes[c.get_id()] = n_res
    
    sorted_chains = sorted(chain_sizes.items(), key=lambda x: x[1], reverse=True)
    receptor_chain = sorted_chains[0][0]
    ligand_chain = sorted_chains[1][0] if len(sorted_chains) > 1 else None
    
    print(f"  Receptor chain: {receptor_chain} ({chain_sizes[receptor_chain]} residues)")
    if ligand_chain:
        print(f"  Ligand chain:   {ligand_chain} ({chain_sizes[ligand_chain]} residues)")
    
    # Save receptor
    receptor_path = os.path.join(output_dir, "receptor.pdb")
    io.save(receptor_path, ChainSelect(receptor_chain))
    result["receptor"] = receptor_path
    
    # Save ligand
    if ligand_chain:
        ligand_path = os.path.join(output_dir, "ligand.pdb")
        io.save(ligand_path, ChainSelect(ligand_chain))
        result["ligand"] = ligand_path
    
    # Save full complex
    complex_path = os.path.join(output_dir, "af3_complex.pdb")
    io.save(complex_path)
    result["complex"] = complex_path
    
    return result


def run_dockgraph_inference(receptor_path: str, ligand_path: str,
                            model_path: str, output_path: str,
                            device: str = "cuda") -> dict:
    """
    Run DockGraph inference to refine the docking pose.
    
    DockGraph is a dual-encoder GNN that predicts a rigid-body transformation
    (rotation θ ∈ SO(3), translation t ∈ ℝ³) to refine the ligand position 
    relative to the receptor, along with a confidence score c ∈ [0,1].
    
    The model processes:
      - Receptor graph: Cα contact graph with 26D node features per residue
      - Ligand graph:   Same representation for the mobile chain
      - Cross-graph:    Inter-chain edges at the interface
    
    Parameters
    ----------
    receptor_path : str
        Path to receptor PDB file
    ligand_path : str  
        Path to ligand PDB file
    model_path : str
        Path to DockGraph TorchScript model (best_model.pt)
    output_path : str
        Path to save the refined complex PDB
    device : str
        'cuda' or 'cpu'
        
    Returns
    -------
    dict
        Prediction results: rotation, translation, confidence, output_path
    """
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: PyTorch required for DockGraph inference")
    
    # Check model file
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"Expected: DockGraph best_model.pt (epoch 42)")
        print(f"Location on server: experiments/20251126_185619_full_training/best_model.pt")
        sys.exit(1)
    
    print(f"\n[DockGraph Inference]")
    print(f"  Receptor: {receptor_path}")
    print(f"  Ligand:   {ligand_path}")
    print(f"  Model:    {model_path}")
    print(f"  Device:   {device}")
    
    # Load model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    # Build graph features from PDB structures
    # This uses the same featurization as in production/predict.py
    from utils.graph_builder import build_protein_graph
    from utils.feature_extractor import extract_node_features
    
    # Build receptor graph
    rec_graph = build_protein_graph(receptor_path)
    rec_features = extract_node_features(receptor_path)
    
    # Build ligand graph  
    lig_graph = build_protein_graph(ligand_path)
    lig_features = extract_node_features(ligand_path)
    
    # Forward pass
    with torch.no_grad():
        # Model outputs: rotation (Rodrigues), translation, confidence
        rotation, translation, confidence = model(
            rec_features.to(device), rec_graph.to(device),
            lig_features.to(device), lig_graph.to(device)
        )
    
    # Convert Rodrigues rotation to rotation matrix
    theta = rotation.cpu().numpy()
    angle = np.linalg.norm(theta)
    if angle > 1e-8:
        axis = theta / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    else:
        R = np.eye(3)
    
    t = translation.cpu().numpy()
    conf = confidence.item()
    
    print(f"\n  Predicted transformation:")
    print(f"    Rotation angle: {np.degrees(angle):.2f}°")
    print(f"    Translation:    [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] Å")
    print(f"    Confidence:     {conf:.4f}")
    
    # Apply transformation to ligand and save refined complex
    apply_transformation(receptor_path, ligand_path, R, t, output_path)
    
    return {
        "rotation_angle_deg": float(np.degrees(angle)),
        "translation_angstrom": t.tolist(),
        "confidence": float(conf),
        "output_path": output_path,
    }


def apply_transformation(receptor_path: str, ligand_path: str,
                         R: np.ndarray, t: np.ndarray, output_path: str):
    """
    Apply rigid-body transformation to ligand and combine with receptor.
    
    The transformation is applied to the ligand Cα centroid frame:
      x' = R @ (x - centroid) + centroid + t
    
    Parameters
    ----------
    receptor_path : str
        Path to receptor PDB
    ligand_path : str
        Path to ligand PDB  
    R : np.ndarray
        3x3 rotation matrix ∈ SO(3)
    t : np.ndarray
        Translation vector ∈ ℝ³ (Angstroms)
    output_path : str
        Path to save combined refined complex
    """
    from Bio.PDB import PDBParser, PDBIO
    
    parser = PDBParser(QUIET=True)
    
    # Load structures
    rec_struct = parser.get_structure("receptor", receptor_path)
    lig_struct = parser.get_structure("ligand", ligand_path)
    
    # Compute ligand centroid (Cα atoms)
    ca_coords = []
    for atom in lig_struct.get_atoms():
        if atom.get_name() == "CA":
            ca_coords.append(atom.get_vector().get_array())
    centroid = np.mean(ca_coords, axis=0)
    
    # Apply transformation to all ligand atoms
    for atom in lig_struct.get_atoms():
        coord = atom.get_vector().get_array()
        new_coord = R @ (coord - centroid) + centroid + t
        atom.set_coord(new_coord)
    
    # Combine receptor and transformed ligand
    # Add ligand chain to receptor structure
    rec_model = rec_struct[0]
    lig_model = lig_struct[0]
    
    for chain in lig_model.get_chains():
        rec_model.add(chain.copy())
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    io = PDBIO()
    io.set_structure(rec_struct)
    io.save(output_path)
    print(f"  [OK] Refined complex saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DockGraph docking refinement on CASP15 targets"
    )
    parser.add_argument(
        "--af3_output_dir", type=str, required=True,
        help="Directory containing AlphaFold3 prediction output"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to DockGraph TorchScript model (best_model.pt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="predictions",
        help="Output directory for refined structures"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--af3_complex_pdb", type=str, default=None,
        help="Directly provide a PDB file instead of parsing AF3 output dir"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.af3_complex_pdb:
        # Direct PDB input mode
        print(f"Using direct PDB input: {args.af3_complex_pdb}")
        af3_result = parse_af3_output(
            os.path.dirname(args.af3_complex_pdb), 
            os.path.join(args.output_dir, "parsed")
        )
    else:
        # Parse AF3 output directory
        print(f"Parsing AlphaFold3 output from: {args.af3_output_dir}")
        af3_result = parse_af3_output(
            args.af3_output_dir, 
            os.path.join(args.output_dir, "parsed")
        )
    
    # Run DockGraph
    output_path = os.path.join(args.output_dir, "refined_complex.pdb")
    prediction = run_dockgraph_inference(
        receptor_path=af3_result["receptor"],
        ligand_path=af3_result["ligand"],
        model_path=args.model_path,
        output_path=output_path,
        device=args.device,
    )
    
    # Save prediction summary
    summary = {
        "af3_confidence": af3_result.get("confidence", {}),
        "dockgraph_prediction": prediction,
    }
    summary_path = os.path.join(args.output_dir, "prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Prediction summary saved to {summary_path}")


if __name__ == "__main__":
    main()
