#!/usr/bin/env python3
"""
evaluate_casp15.py
==================
Evaluate DockGraph predictions against CASP15 ground truth structures.

Computes all standard protein-protein docking metrics:
  - DockQ score: Combined metric ∈ [0,1] (Basu & Wallner, PLoS ONE 2016)
  - I-RMSD:     Interface RMSD (Å) - backbone atoms at interface
  - L-RMSD:     Ligand RMSD (Å) - backbone after receptor superposition  
  - fnat:       Fraction of native contacts recovered
  - CAPRI quality classification: Incorrect/Acceptable/Medium/High

Mathematically:
  DockQ = (fnat + 1/(1+(I-RMSD/1.5)²) + 1/(1+(L-RMSD/8.5)²)) / 3

CAPRI classification thresholds:
  High:       fnat ≥ 0.5  AND (L-RMSD ≤ 1.0 OR I-RMSD ≤ 1.0)
  Medium:     fnat ≥ 0.3  AND (L-RMSD ≤ 5.0 OR I-RMSD ≤ 2.0)  
  Acceptable: fnat ≥ 0.1  AND (L-RMSD ≤ 10.0 OR I-RMSD ≤ 4.0)
  Incorrect:  Everything else

DockQ classification:
  High:       DockQ ≥ 0.80
  Medium:     0.49 < DockQ < 0.80
  Acceptable: 0.23 ≤ DockQ ≤ 0.49
  Incorrect:  DockQ < 0.23

Usage:
    python evaluate_casp15.py \
        --prediction predictions/H1141_T206/refined_complex.pdb \
        --ground_truth ground_truth/H1141/9ERU.pdb \
        --receptor_chain A --ligand_chain B \
        --output_dir results/H1141_T206/

References:
    [1] Basu & Wallner. "DockQ: A Quality Measure for Protein-Protein 
        Docking Models." PLoS ONE 11(8), 2016.
    [2] Lensink et al. "Prediction of homoprotein and heteroprotein 
        complexes by protein docking and template-based modeling: 
        A CASP-CAPRI experiment." Proteins 84, 2016.
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
from pathlib import Path

try:
    from Bio.PDB import PDBParser, MMCIFParser, Superimposer
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    sys.exit("ERROR: BioPython required. Install: pip install biopython")

warnings.filterwarnings("ignore")


# =============================================================================
# Contact & Interface Definitions
# =============================================================================

# Contact distance threshold (Å) - standard CAPRI definition
CONTACT_THRESHOLD = 5.0

# Interface residue distance threshold (Å) - any heavy atom within this distance
INTERFACE_THRESHOLD = 10.0

# Backbone atom names for RMSD calculations
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# =============================================================================
# Core Metric Functions
# =============================================================================

def get_residue_contacts(chain1, chain2, threshold=CONTACT_THRESHOLD):
    """
    Compute residue-residue contacts between two chains.
    
    A contact is defined as any pair of residues where at least one pair
    of heavy atoms is within the distance threshold. This follows the 
    standard CAPRI contact definition.
    
    Parameters
    ----------
    chain1 : Bio.PDB.Chain
    chain2 : Bio.PDB.Chain
    threshold : float
        Distance cutoff in Angstroms
        
    Returns
    -------
    set of tuple
        Set of (res1_id, res2_id) contact pairs, where res_id = (chain_id, resseq, icode)
    """
    contacts = set()
    
    residues1 = [r for r in chain1.get_residues() if is_aa(r, standard=True)]
    residues2 = [r for r in chain2.get_residues() if is_aa(r, standard=True)]
    
    for r1 in residues1:
        for r2 in residues2:
            # Check if any heavy atom pair is within threshold
            for a1 in r1.get_atoms():
                if a1.element == 'H':
                    continue
                for a2 in r2.get_atoms():
                    if a2.element == 'H':
                        continue
                    dist = a1 - a2  # BioPython distance operator
                    if dist < threshold:
                        r1_id = (r1.get_parent().get_id(), r1.get_id()[1], r1.get_id()[2])
                        r2_id = (r2.get_parent().get_id(), r2.get_id()[1], r2.get_id()[2])
                        contacts.add((r1_id, r2_id))
                        break
                else:
                    continue
                break
    
    return contacts


def compute_fnat(native_contacts, model_contacts):
    """
    Compute fraction of native contacts (fnat).
    
    fnat = |native_contacts ∩ model_contacts| / |native_contacts|
    
    Parameters
    ----------
    native_contacts : set
        Contact pairs from ground truth
    model_contacts : set
        Contact pairs from prediction
        
    Returns
    -------
    float
        fnat ∈ [0, 1]
    """
    if len(native_contacts) == 0:
        return 0.0
    
    recovered = native_contacts.intersection(model_contacts)
    return len(recovered) / len(native_contacts)


def get_interface_residues(chain1, chain2, threshold=INTERFACE_THRESHOLD):
    """
    Identify interface residues based on inter-chain distance.
    
    A residue is at the interface if any of its heavy atoms is within
    the threshold distance from any heavy atom of the partner chain.
    
    Returns
    -------
    tuple of (set, set)
        Interface residue IDs for chain1 and chain2
    """
    interface1 = set()
    interface2 = set()
    
    residues1 = [r for r in chain1.get_residues() if is_aa(r, standard=True)]
    residues2 = [r for r in chain2.get_residues() if is_aa(r, standard=True)]
    
    for r1 in residues1:
        for r2 in residues2:
            for a1 in r1.get_atoms():
                if a1.element == 'H':
                    continue
                for a2 in r2.get_atoms():
                    if a2.element == 'H':
                        continue
                    if a1 - a2 < threshold:
                        interface1.add(r1.get_id()[1])
                        interface2.add(r2.get_id()[1])
                        break
                else:
                    continue
                break
    
    return interface1, interface2


def get_backbone_atoms(chain, residue_ids=None):
    """
    Extract backbone atom coordinates from a chain.
    
    Parameters
    ----------
    chain : Bio.PDB.Chain
    residue_ids : set or None
        If provided, only extract from these residue sequence numbers.
        If None, extract from all standard residues.
        
    Returns
    -------
    tuple of (list, np.ndarray)
        (atom_labels, coordinates) where atom_labels help with matching
    """
    atoms = []
    coords = []
    
    for residue in chain.get_residues():
        if not is_aa(residue, standard=True):
            continue
        if residue_ids is not None and residue.get_id()[1] not in residue_ids:
            continue
        
        for atom_name in ["N", "CA", "C", "O"]:
            if atom_name in residue:
                atom = residue[atom_name]
                label = (residue.get_id()[1], atom_name)
                atoms.append(label)
                coords.append(atom.get_vector().get_array())
    
    return atoms, np.array(coords) if coords else np.array([]).reshape(0, 3)


def compute_irmsd(native_rec_chain, native_lig_chain, 
                  model_rec_chain, model_lig_chain,
                  interface_threshold=INTERFACE_THRESHOLD):
    """
    Compute Interface RMSD (I-RMSD).
    
    Procedure:
    1. Identify interface residues in the native complex
    2. Extract backbone atoms at the interface (both chains)
    3. Superpose model interface onto native interface
    4. Compute RMSD of superposed backbone atoms
    
    I-RMSD measures how well the model reproduces the native interface 
    geometry, independent of the overall chain orientations.
    
    Parameters
    ----------
    native_rec_chain, native_lig_chain : Bio.PDB.Chain
        Chains from ground truth structure
    model_rec_chain, model_lig_chain : Bio.PDB.Chain
        Chains from predicted structure
        
    Returns
    -------
    float
        I-RMSD in Angstroms
    """
    # Get interface residues from native
    nat_iface_rec, nat_iface_lig = get_interface_residues(
        native_rec_chain, native_lig_chain, interface_threshold
    )
    
    if not nat_iface_rec and not nat_iface_lig:
        print("  WARNING: No interface residues found in native structure")
        return float('inf')
    
    # Get backbone atoms at interface
    nat_rec_labels, nat_rec_coords = get_backbone_atoms(native_rec_chain, nat_iface_rec)
    nat_lig_labels, nat_lig_coords = get_backbone_atoms(native_lig_chain, nat_iface_lig)
    
    mod_rec_labels, mod_rec_coords = get_backbone_atoms(model_rec_chain, nat_iface_rec)
    mod_lig_labels, mod_lig_coords = get_backbone_atoms(model_lig_chain, nat_iface_lig)
    
    # Match atoms between native and model
    nat_combined, mod_combined = match_atoms(
        nat_rec_labels + nat_lig_labels,
        np.vstack([nat_rec_coords, nat_lig_coords]) if len(nat_rec_coords) > 0 and len(nat_lig_coords) > 0 else nat_rec_coords,
        mod_rec_labels + mod_lig_labels,
        np.vstack([mod_rec_coords, mod_lig_coords]) if len(mod_rec_coords) > 0 and len(mod_lig_coords) > 0 else mod_rec_coords,
    )
    
    if len(nat_combined) < 3:
        print("  WARNING: Insufficient matched atoms for I-RMSD calculation")
        return float('inf')
    
    # Superpose and compute RMSD
    sup = Superimposer()
    sup.set_atoms(
        [AtomWrapper(c) for c in nat_combined],
        [AtomWrapper(c) for c in mod_combined]
    )
    return sup.rms


def compute_lrmsd(native_rec_chain, native_lig_chain,
                  model_rec_chain, model_lig_chain):
    """
    Compute Ligand RMSD (L-RMSD).
    
    Procedure:
    1. Superpose model receptor onto native receptor (backbone atoms)
    2. Apply the same transformation to the model ligand
    3. Compute RMSD between transformed model ligand and native ligand backbone
    
    L-RMSD measures how well the model places the ligand relative to the 
    receptor, after removing receptor alignment differences.
    
    Parameters
    ----------
    native_rec_chain, native_lig_chain : Bio.PDB.Chain
        Chains from ground truth structure
    model_rec_chain, model_lig_chain : Bio.PDB.Chain
        Chains from predicted structure
        
    Returns
    -------
    float
        L-RMSD in Angstroms
    """
    # Get all backbone atoms from receptor
    nat_rec_labels, nat_rec_coords = get_backbone_atoms(native_rec_chain)
    mod_rec_labels, mod_rec_coords = get_backbone_atoms(model_rec_chain)
    
    # Match receptor atoms
    nat_rec_matched, mod_rec_matched = match_atoms(
        nat_rec_labels, nat_rec_coords,
        mod_rec_labels, mod_rec_coords
    )
    
    if len(nat_rec_matched) < 3:
        return float('inf')
    
    # Compute receptor superposition
    sup = Superimposer()
    sup.set_atoms(
        [AtomWrapper(c) for c in nat_rec_matched],
        [AtomWrapper(c) for c in mod_rec_matched]
    )
    
    # Get rotation and translation from superposition
    rot, tran = sup.rotran
    
    # Get ligand backbone atoms
    nat_lig_labels, nat_lig_coords = get_backbone_atoms(native_lig_chain)
    mod_lig_labels, mod_lig_coords = get_backbone_atoms(model_lig_chain)
    
    # Match ligand atoms
    nat_lig_matched, mod_lig_matched = match_atoms(
        nat_lig_labels, nat_lig_coords,
        mod_lig_labels, mod_lig_coords
    )
    
    if len(nat_lig_matched) < 3:
        return float('inf')
    
    # Apply receptor superposition to model ligand
    mod_lig_transformed = np.dot(mod_lig_matched, rot) + tran
    
    # Compute RMSD
    diff = nat_lig_matched - mod_lig_transformed
    lrmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    
    return lrmsd


def match_atoms(labels1, coords1, labels2, coords2):
    """
    Match atoms between two structures by residue number and atom name.
    
    Returns matched coordinate arrays of equal length.
    """
    label_to_coord1 = {l: c for l, c in zip(labels1, coords1)}
    label_to_coord2 = {l: c for l, c in zip(labels2, coords2)}
    
    common_labels = set(label_to_coord1.keys()) & set(label_to_coord2.keys())
    common_labels = sorted(common_labels)
    
    if not common_labels:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    matched1 = np.array([label_to_coord1[l] for l in common_labels])
    matched2 = np.array([label_to_coord2[l] for l in common_labels])
    
    return matched1, matched2


class AtomWrapper:
    """Wrapper to make coordinate arrays work with BioPython's Superimposer."""
    def __init__(self, coord):
        self._coord = np.array(coord, dtype='d')
    
    def get_vector(self):
        return self
    
    def get_array(self):
        return self._coord


def compute_dockq(fnat, irmsd, lrmsd):
    """
    Compute DockQ score from components.
    
    DockQ = (fnat + 1/(1+(I-RMSD/1.5)²) + 1/(1+(L-RMSD/8.5)²)) / 3
    
    This formula combines three complementary metrics:
    - fnat:   Captures contact recovery (local accuracy)
    - I-RMSD: Captures interface geometry (medium-range accuracy)
    - L-RMSD: Captures overall ligand placement (global accuracy)
    
    The sigmoidal scaling (1/(1+x²)) maps RMSD values to [0,1] with
    inflection points at 1.5 Å (I-RMSD) and 8.5 Å (L-RMSD).
    
    Reference: Basu & Wallner, PLoS ONE 2016
    """
    d1 = 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    d2 = 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    dockq = (fnat + d1 + d2) / 3.0
    return dockq


def classify_capri(fnat, lrmsd, irmsd):
    """
    Classify docking quality using CAPRI criteria.
    
    CAPRI (Critical Assessment of PRediction of Interactions) classification:
    
    High:       fnat ≥ 0.5  AND (L-RMSD ≤ 1.0 Å OR I-RMSD ≤ 1.0 Å)
    Medium:     fnat ≥ 0.3  AND (L-RMSD ≤ 5.0 Å OR I-RMSD ≤ 2.0 Å)
    Acceptable: fnat ≥ 0.1  AND (L-RMSD ≤ 10.0 Å OR I-RMSD ≤ 4.0 Å)
    Incorrect:  Everything else
    
    Note: CAPRI uses OR for L-RMSD/I-RMSD within each quality tier,
    meaning either metric can qualify a model independently.
    """
    if fnat >= 0.5 and (lrmsd <= 1.0 or irmsd <= 1.0):
        return "High"
    elif fnat >= 0.3 and (lrmsd <= 5.0 or irmsd <= 2.0):
        return "Medium"
    elif fnat >= 0.1 and (lrmsd <= 10.0 or irmsd <= 4.0):
        return "Acceptable"
    else:
        return "Incorrect"


def classify_dockq(dockq):
    """
    Classify quality based on DockQ score thresholds.
    
    DockQ ≥ 0.80: High quality
    0.49 < DockQ < 0.80: Medium quality  
    0.23 ≤ DockQ ≤ 0.49: Acceptable quality
    DockQ < 0.23: Incorrect
    """
    if dockq >= 0.80:
        return "High"
    elif dockq > 0.49:
        return "Medium"
    elif dockq >= 0.23:
        return "Acceptable"
    else:
        return "Incorrect"


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def load_structure(filepath):
    """Load PDB or mmCIF structure."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("structure", filepath)


def evaluate(prediction_path: str, ground_truth_path: str,
             receptor_chain_id: str = "A", ligand_chain_id: str = "B",
             output_dir: str = None) -> dict:
    """
    Full evaluation of a docking prediction against ground truth.
    
    Parameters
    ----------
    prediction_path : str
        Path to predicted complex (PDB or mmCIF)
    ground_truth_path : str
        Path to ground truth complex (PDB or mmCIF)
    receptor_chain_id : str
        Chain ID of the receptor (antigen) in both structures
    ligand_chain_id : str
        Chain ID of the ligand (nanobody) in both structures
    output_dir : str
        Directory to save evaluation results
        
    Returns
    -------
    dict
        Complete evaluation metrics
    """
    print("=" * 70)
    print("DockGraph CASP15 Evaluation")
    print("=" * 70)
    print(f"  Prediction:   {prediction_path}")
    print(f"  Ground truth: {ground_truth_path}")
    print(f"  Receptor:     chain {receptor_chain_id}")
    print(f"  Ligand:       chain {ligand_chain_id}")
    print("=" * 70)
    
    # Load structures
    print("\n[Step 1] Loading structures...")
    pred_struct = load_structure(prediction_path)
    gt_struct = load_structure(ground_truth_path)
    
    # Get chains
    pred_model = pred_struct[0]
    gt_model = gt_struct[0]
    
    # List available chains
    pred_chains = [c.get_id() for c in pred_model.get_chains()]
    gt_chains = [c.get_id() for c in gt_model.get_chains()]
    print(f"  Prediction chains: {pred_chains}")
    print(f"  Ground truth chains: {gt_chains}")
    
    try:
        pred_rec = pred_model[receptor_chain_id]
        pred_lig = pred_model[ligand_chain_id]
        gt_rec = gt_model[receptor_chain_id]
        gt_lig = gt_model[ligand_chain_id]
    except KeyError as e:
        print(f"\nERROR: Chain {e} not found!")
        print(f"  Available in prediction: {pred_chains}")
        print(f"  Available in ground truth: {gt_chains}")
        print(f"\n  Hint: AF3 may use different chain IDs than the PDB.")
        print(f"  Try mapping: prediction chain -> ground truth chain")
        sys.exit(1)
    
    # Report chain sizes
    for label, rec, lig in [("Ground truth", gt_rec, gt_lig), ("Prediction", pred_rec, pred_lig)]:
        n_rec = sum(1 for r in rec.get_residues() if is_aa(r, standard=True))
        n_lig = sum(1 for r in lig.get_residues() if is_aa(r, standard=True))
        print(f"  {label}: receptor={n_rec}aa, ligand={n_lig}aa")
    
    # Step 2: Compute native contacts
    print("\n[Step 2] Computing native contacts...")
    native_contacts = get_residue_contacts(gt_rec, gt_lig, CONTACT_THRESHOLD)
    model_contacts = get_residue_contacts(pred_rec, pred_lig, CONTACT_THRESHOLD)
    print(f"  Native contacts:  {len(native_contacts)}")
    print(f"  Model contacts:   {len(model_contacts)}")
    
    # Step 3: Compute fnat
    print("\n[Step 3] Computing fnat...")
    fnat = compute_fnat(native_contacts, model_contacts)
    print(f"  fnat = {fnat:.4f}")
    
    # Step 4: Compute I-RMSD
    print("\n[Step 4] Computing Interface RMSD...")
    irmsd = compute_irmsd(gt_rec, gt_lig, pred_rec, pred_lig)
    print(f"  I-RMSD = {irmsd:.4f} Å")
    
    # Step 5: Compute L-RMSD
    print("\n[Step 5] Computing Ligand RMSD...")
    lrmsd = compute_lrmsd(gt_rec, gt_lig, pred_rec, pred_lig)
    print(f"  L-RMSD = {lrmsd:.4f} Å")
    
    # Step 6: Compute DockQ
    print("\n[Step 6] Computing DockQ score...")
    dockq = compute_dockq(fnat, irmsd, lrmsd)
    print(f"  DockQ = {dockq:.4f}")
    
    # Step 7: Classifications
    capri_class = classify_capri(fnat, lrmsd, irmsd)
    dockq_class = classify_dockq(dockq)
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  fnat:              {fnat:.4f}")
    print(f"  I-RMSD:            {irmsd:.4f} Å")
    print(f"  L-RMSD:            {lrmsd:.4f} Å")
    print(f"  DockQ:             {dockq:.4f}")
    print(f"  CAPRI class:       {capri_class}")
    print(f"  DockQ class:       {dockq_class}")
    print(f"  Native contacts:   {len(native_contacts)}")
    print(f"  Recovered:         {len(native_contacts & model_contacts)}")
    print(f"{'=' * 70}")
    
    # Compile results
    results = {
        "prediction_file": prediction_path,
        "ground_truth_file": ground_truth_path,
        "receptor_chain": receptor_chain_id,
        "ligand_chain": ligand_chain_id,
        "metrics": {
            "fnat": round(fnat, 4),
            "irmsd": round(irmsd, 4),
            "lrmsd": round(lrmsd, 4),
            "dockq": round(dockq, 4),
        },
        "classification": {
            "capri": capri_class,
            "dockq": dockq_class,
        },
        "contacts": {
            "native": len(native_contacts),
            "model": len(model_contacts),
            "recovered": len(native_contacts & model_contacts),
        },
        "success": dockq >= 0.23,  # Binary success threshold
    }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DockGraph predictions against CASP15 ground truth"
    )
    parser.add_argument(
        "--prediction", type=str, required=True,
        help="Path to predicted complex structure (PDB or mmCIF)"
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True,
        help="Path to ground truth structure (PDB or mmCIF)"
    )
    parser.add_argument(
        "--receptor_chain", type=str, default="A",
        help="Chain ID of receptor/antigen (default: A)"
    )
    parser.add_argument(
        "--ligand_chain", type=str, default="B",
        help="Chain ID of ligand/nanobody (default: B)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    results = evaluate(
        prediction_path=args.prediction,
        ground_truth_path=args.ground_truth,
        receptor_chain_id=args.receptor_chain,
        ligand_chain_id=args.ligand_chain,
        output_dir=args.output_dir,
    )
    
    return results


if __name__ == "__main__":
    main()
