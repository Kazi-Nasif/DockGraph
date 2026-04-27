#!/usr/bin/env python3
"""
download_ground_truth.py
========================
Download and prepare ground truth crystal structures for CASP15 targets.

For H1141 (CASP15 T206): PDB 9ERU - Mouse CNPase catalytic domain with nanobody 7E
  - Chain A: CNPase catalytic domain (mouse, 219 residues)
  - Chain B: Nanobody 7E (alpaca, 127 residues)
  - Method: X-ray crystallography
  - Reference: Kursula et al., "Nanobodies against the myelin enzyme CNPase 
    as tools for structural and functional studies"

This script downloads the structure from RCSB PDB in both PDB and mmCIF formats,
extracts individual chains, and prepares them for DockGraph evaluation.

Usage:
    python download_ground_truth.py --target H1141 --output_dir ground_truth/
"""

import os
import sys
import argparse
import urllib.request
import json
from pathlib import Path

# =============================================================================
# CASP15 Target Registry
# =============================================================================
# Maps CASP15 target IDs to their released PDB structures.
# Only targets with released crystal structures are included.
# These are nanobody-antigen complexes used by AlphaRED for CASP15 evaluation.
#
# CASP Target -> PDB ID mapping sourced from:
#   - CASP15 official results: https://predictioncenter.org/casp15/
#   - Kursula et al., bioRxiv/J Neurochem (CNPase nanobody series)
#   - DMFold CASP15 paper (Zheng et al.)
# =============================================================================

CASP15_TARGETS = {
    "H1141": {
        "pdb_id": "9ERU",
        "description": "Mouse CNPase catalytic domain with nanobody 7E",
        "casp_target": "T206",
        "organism": "mouse/alpaca",
        "method": "X-RAY",
        "chains": {
            "A": {"name": "CNPase catalytic domain", "organism": "Mus musculus", "type": "antigen"},
            "B": {"name": "Nanobody 7E", "organism": "Vicugna pacos", "type": "nanobody"},
        },
        "complex_type": "nanobody-antigen",
        "residues": 346,
    },
    # Other CASP15 nanobody-antigen targets for future expansion:
    # H1140: CNPase + another nanobody
    # H1142: CNPase + nanobody 8C -> PDB 9ERW
    # H1144: CNPase + nanobody variant
}


def download_pdb(pdb_id: str, output_dir: str, fmt: str = "pdb") -> str:
    """
    Download structure file from RCSB PDB.
    
    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier
    output_dir : str
        Directory to save downloaded file
    fmt : str
        Format: 'pdb' for PDB format, 'cif' for mmCIF format
        
    Returns
    -------
    str
        Path to downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if fmt == "pdb":
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        filename = f"{pdb_id.upper()}.pdb"
    elif fmt == "cif":
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        filename = f"{pdb_id.upper()}.cif"
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'pdb' or 'cif'.")
    
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        print(f"  [SKIP] {filepath} already exists")
        return filepath
    
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  [OK] Saved to {filepath}")
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: Could not download {url}")
        print(f"  Note: PDB 9ERU may not yet be in PDB format. Trying mmCIF...")
        if fmt == "pdb":
            return download_pdb(pdb_id, output_dir, fmt="cif")
        sys.exit(1)
    
    return filepath


def extract_chains_biopython(structure_path: str, output_dir: str, chains_info: dict) -> dict:
    """
    Extract individual chains from the complex structure using BioPython.
    
    This creates separate PDB files for each chain (receptor and ligand),
    which is the format DockGraph expects for input.
    
    Parameters
    ----------
    structure_path : str
        Path to the downloaded complex structure
    output_dir : str
        Directory for extracted chain files
    chains_info : dict
        Chain metadata from CASP15_TARGETS
        
    Returns
    -------
    dict
        Mapping of chain_id -> extracted file path
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
    except ImportError:
        print("  [WARN] BioPython not installed. Install with: pip install biopython")
        print("  Skipping chain extraction.")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect format from extension
    ext = os.path.splitext(structure_path)[1].lower()
    if ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("complex", structure_path)
    model = structure[0]
    
    class ChainSelect(Select):
        """Select a specific chain for output."""
        def __init__(self, chain_id):
            self.chain_id = chain_id
        def accept_chain(self, chain):
            return chain.get_id() == self.chain_id
    
    extracted = {}
    io = PDBIO()
    io.set_structure(structure)
    
    for chain_id, info in chains_info.items():
        if chain_id in [c.get_id() for c in model.get_chains()]:
            chain_file = os.path.join(output_dir, f"chain_{chain_id}_{info['type']}.pdb")
            io.save(chain_file, ChainSelect(chain_id))
            
            # Count residues
            chain = model[chain_id]
            n_res = sum(1 for r in chain.get_residues() 
                       if r.get_id()[0] == ' ')  # Standard residues only
            
            print(f"  [OK] Chain {chain_id} ({info['name']}): {n_res} residues -> {chain_file}")
            extracted[chain_id] = chain_file
        else:
            print(f"  [WARN] Chain {chain_id} not found in structure")
            # List available chains
            available = [c.get_id() for c in model.get_chains()]
            print(f"  Available chains: {available}")
    
    return extracted


def verify_sequence_match(structure_path: str, target_sequences: dict) -> dict:
    """
    Verify that the crystal structure sequences match the CASP15 target sequences.
    
    This is critical for ensuring our evaluation is comparing the correct structures.
    Sequence differences could indicate:
      1. Different chain assignments
      2. Expression tags or truncations in the crystal structure
      3. Post-translational modifications
    
    Parameters
    ----------
    structure_path : str
        Path to the ground truth structure
    target_sequences : dict
        Chain ID -> expected sequence from CASP target definition
        
    Returns
    -------
    dict
        Chain ID -> {matched: bool, identity: float, alignment_info: str}
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        from Bio.PDB.Polypeptide import PPBuilder
    except ImportError:
        print("  [WARN] BioPython not available for sequence verification")
        return {}
    
    ext = os.path.splitext(structure_path)[1].lower()
    if ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("complex", structure_path)
    ppb = PPBuilder()
    
    results = {}
    for chain in structure[0].get_chains():
        chain_id = chain.get_id()
        # Build polypeptide sequence from structure
        pp_list = ppb.build_peptides(chain)
        struct_seq = "".join(str(pp.get_sequence()) for pp in pp_list)
        
        if chain_id in target_sequences:
            target_seq = target_sequences[chain_id]
            
            # Simple identity check (structure may have missing residues)
            # We check if the structure sequence is a subsequence or has high overlap
            matches = 0
            min_len = min(len(struct_seq), len(target_seq))
            for i in range(min_len):
                if struct_seq[i] == target_seq[i]:
                    matches += 1
            
            identity = matches / max(len(struct_seq), len(target_seq)) if max(len(struct_seq), len(target_seq)) > 0 else 0
            
            results[chain_id] = {
                "matched": identity > 0.90,
                "identity": identity,
                "struct_length": len(struct_seq),
                "target_length": len(target_seq),
                "struct_seq_preview": struct_seq[:50] + "..." if len(struct_seq) > 50 else struct_seq,
                "target_seq_preview": target_seq[:50] + "..." if len(target_seq) > 50 else target_seq,
            }
            
            status = "MATCH" if identity > 0.90 else "MISMATCH"
            print(f"  Chain {chain_id}: {status} (identity={identity:.1%}, "
                  f"struct={len(struct_seq)}aa, target={len(target_seq)}aa)")
            
            if identity <= 0.90:
                print(f"    WARNING: Low sequence identity! Check chain assignment.")
                print(f"    Structure: {struct_seq[:60]}...")
                print(f"    Target:    {target_seq[:60]}...")
        else:
            print(f"  Chain {chain_id}: Not in target (length={len(struct_seq)}aa) - may be crystallographic artifact")
    
    return results


def prepare_ground_truth(target_id: str, output_dir: str):
    """
    Full pipeline: download, extract, verify ground truth for a CASP15 target.
    """
    if target_id not in CASP15_TARGETS:
        print(f"ERROR: Unknown target '{target_id}'")
        print(f"Available targets: {list(CASP15_TARGETS.keys())}")
        sys.exit(1)
    
    target = CASP15_TARGETS[target_id]
    pdb_id = target["pdb_id"]
    
    print(f"=" * 70)
    print(f"CASP15 Ground Truth Preparation")
    print(f"=" * 70)
    print(f"  Target:      {target_id} ({target['casp_target']})")
    print(f"  PDB ID:      {pdb_id}")
    print(f"  Description: {target['description']}")
    print(f"  Organism:    {target['organism']}")
    print(f"  Method:      {target['method']}")
    print(f"  Complex:     {target['complex_type']}")
    print(f"  Residues:    {target['residues']}")
    print(f"=" * 70)
    
    # Step 1: Download structure
    print(f"\n[Step 1] Downloading crystal structure from RCSB PDB...")
    gt_dir = os.path.join(output_dir, target_id)
    structure_path = download_pdb(pdb_id, gt_dir, fmt="cif")
    
    # Also try PDB format
    pdb_path = download_pdb(pdb_id, gt_dir, fmt="pdb")
    
    # Step 2: Extract individual chains
    print(f"\n[Step 2] Extracting individual chains...")
    chains_dir = os.path.join(gt_dir, "chains")
    # Use whichever format was successfully downloaded
    use_path = pdb_path if os.path.exists(pdb_path) and pdb_path.endswith('.pdb') else structure_path
    extracted = extract_chains_biopython(use_path, chains_dir, target["chains"])
    
    # Step 3: Verify sequence match
    print(f"\n[Step 3] Verifying sequences against CASP15 target definition...")
    # Expected sequences from the CASP15 target
    target_sequences = {
        "A": "GLEKDFLPLYFGWFLTKKSSETLRKAGQVFLEELGNHKAFKKELRHFISGDEPKEKLELVSYFGKRPPGVLHCTTKFCDYKAAGAEEYAQQEVVKRSYGKAFKLSISALFVTPKTAGAQVVLTDQELQLWPSDLDKPSASEGLPPGSRAHVTLGCAADVQPVQTGLDLLDILQQVKGGSQGEAVGELPRGKLYSLGKGRWMLSLTKKMEVKAIFTGYYG",
        "B": "EVQLEESGGGWVHPGGSLRLSCAASGNVFGVNTMAWYRQAPGKQREQRELVASITDYGTTEYADSVKGRFTISGDNAKATVYLQMNSLKPEDTAVYYCNMDLTVMTATSSLYAYDYWGQGTQVTVSS",
    }
    seq_results = verify_sequence_match(use_path, target_sequences)
    
    # Step 4: Save metadata
    print(f"\n[Step 4] Saving metadata...")
    metadata = {
        "target_id": target_id,
        "casp_target": target["casp_target"],
        "pdb_id": pdb_id,
        "description": target["description"],
        "complex_type": target["complex_type"],
        "structure_file": os.path.basename(structure_path),
        "chains": target["chains"],
        "extracted_chains": {k: os.path.basename(v) for k, v in extracted.items()},
        "sequence_verification": seq_results,
        "notes": [
            "Chain A = CNPase catalytic domain (receptor/antigen)",
            "Chain B = Nanobody 7E (ligand/nanobody)",
            "For DockQ evaluation: receptor=A, ligand=B",
            "Crystal structure from Kursula et al. nanobody-CNPase series",
        ],
    }
    
    meta_path = os.path.join(gt_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Metadata saved to {meta_path}")
    
    print(f"\n{'=' * 70}")
    print(f"Ground truth preparation complete!")
    print(f"  Complex structure: {structure_path}")
    print(f"  Individual chains: {chains_dir}/")
    print(f"  Metadata:          {meta_path}")
    print(f"{'=' * 70}")
    
    return gt_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare CASP15 ground truth structures for DockGraph evaluation"
    )
    parser.add_argument(
        "--target", type=str, default="H1141",
        help="CASP15 target ID (default: H1141)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="ground_truth",
        help="Output directory (default: ground_truth/)"
    )
    parser.add_argument(
        "--list_targets", action="store_true",
        help="List available CASP15 targets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_targets:
        print("Available CASP15 targets:")
        for tid, info in CASP15_TARGETS.items():
            print(f"  {tid} ({info['casp_target']}): {info['description']} [PDB: {info['pdb_id']}]")
        return
    
    prepare_ground_truth(args.target, args.output_dir)


if __name__ == "__main__":
    main()
