#!/usr/bin/env python3
"""Step 2B: Understand PDB File Structure"""

import os
from pathlib import Path
from Bio import PDB
import numpy as np

# Your AlphaRED path
ALPHARED_BENCHMARK = "/mnt/bst/a100/bdeng2/knasif/Protein Docking/DockGraph/data/benchmark"

def examine_single_target(target_name="1A2K", difficulty="rigid_targets"):
    """Examine a single target"""
    print("="*80)
    print(f"EXAMINING TARGET: {target_name} ({difficulty})")
    print("="*80)
    
    target_dir = Path(ALPHARED_BENCHMARK) / difficulty / target_name
    
    if not target_dir.exists():
        print(f"❌ Directory not found: {target_dir}")
        return None
    
    print(f"\n✓ Target directory: {target_dir}")
    
    # List all files
    print(f"\nFiles in {target_name}/:")
    files = sorted(os.listdir(target_dir))
    for f in files:
        file_path = target_dir / f
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {f:<20} ({size:.1f} KB)")
    
    # Parse PDB files
    parser = PDB.PDBParser(QUIET=True)
    pdb_files = [f for f in files if f.endswith('.pdb')]
    
    print(f"\n{'='*80}")
    print("PDB FILE ANALYSIS")
    print(f"{'='*80}")
    
    for pdb_file in pdb_files:
        pdb_path = target_dir / pdb_file
        file_key = pdb_file.replace('.pdb', '')
        
        print(f"\n--- {pdb_file} ---")
        
        try:
            structure = parser.get_structure(file_key, pdb_path)
            models = list(structure.get_models())
            print(f"  Models: {len(models)}")
            
            for model in models[:1]:
                chains = list(model.get_chains())
                print(f"  Chains: {len(chains)} - {[c.id for c in chains]}")
                
                total_residues = sum(len(list(chain.get_residues())) for chain in chains)
                print(f"  Total residues: {total_residues}")
                
                atoms = list(model.get_atoms())
                coords = np.array([atom.get_coord() for atom in atoms])
                print(f"  Total atoms: {len(atoms)}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Read partners file
    partners_file = target_dir / "partners"
    if partners_file.exists():
        print(f"\n{'='*80}")
        print("PARTNERS FILE")
        print(f"{'='*80}")
        with open(partners_file, 'r') as f:
            print(f.read())
    
    print(f"\n✓ Analysis complete!")
    return True

def quick_check():
    """Quick check"""
    print("QUICK CHECK OF ALPHARED BENCHMARK")
    print("="*80)
    
    benchmark_path = Path(ALPHARED_BENCHMARK)
    
    if not benchmark_path.exists():
        print(f"❌ Path not found: {benchmark_path}")
        return False
    
    print(f"✓ Benchmark path: {benchmark_path}\n")
    
    for difficulty in ['rigid_targets', 'medium_targets', 'difficult_targets']:
        diff_path = benchmark_path / difficulty
        if diff_path.exists():
            targets = [d for d in os.listdir(diff_path) if os.path.isdir(diff_path / d)]
            print(f"  {difficulty}: {len(targets)} targets")
    
    return True

import sys

if __name__ == "__main__":
    import sys
    if quick_check():
        pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1A2K"
        diff = sys.argv[2] if len(sys.argv) > 2 else "rigid_targets"
        print("\n" + "="*80)
        examine_single_target(pdb_id, diff)
