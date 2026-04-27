#!/usr/bin/env python3
"""
Step 3: Simple Data Loader and I-RMSD Calculator
Load protein structures and calculate Interface-RMSD
"""

import numpy as np
from Bio import PDB
from pathlib import Path
from typing import Tuple, List
import sys

# Paths - CORRECTED
PROJECT_ROOT = Path(__file__).parent.parent
ALPHARED_BENCHMARK = Path("/mnt/bst/a100/bdeng2/knasif/Protein Docking/DockGraph/data/benchmark")  # ← FIXED!


class ProteinDockingLoader:
    """Simple loader for protein docking data"""
    
    def __init__(self, benchmark_path=None):
        if benchmark_path is None:
            benchmark_path = ALPHARED_BENCHMARK
        self.benchmark_path = Path(benchmark_path)
        self.parser = PDB.PDBParser(QUIET=True)
        
    def load_target(self, target_name, difficulty="rigid_targets"):
        """Load a single docking target"""
        target_dir = self.benchmark_path / difficulty / target_name
        
        if not target_dir.exists():
            raise FileNotFoundError(f"Target not found: {target_dir}")
        
        # Load structures
        receptor_unbound = self.parser.get_structure(
            f"{target_name}_r_u",
            target_dir / f"{target_name}_r_u.pdb"
        )
        
        ligand_unbound = self.parser.get_structure(
            f"{target_name}_l_u",
            target_dir / f"{target_name}_l_u.pdb"
        )
        
        bound_complex = self.parser.get_structure(
            f"{target_name}_b",
            target_dir / f"{target_name}_b.pdb"
        )
        
        return {
            'target_name': target_name,
            'difficulty': difficulty,
            'receptor_unbound': receptor_unbound,
            'ligand_unbound': ligand_unbound,
            'bound_complex': bound_complex,
        }
    
    def calculate_rmsd(self, coords1, coords2):
        """Calculate RMSD between two sets of coordinates"""
        if len(coords1) != len(coords2):
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]
        
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


if __name__ == "__main__":
    # Test
    print("Testing data loader...")
    loader = ProteinDockingLoader()
    data = loader.load_target("1A2K", "rigid_targets")
    print(f"✓ Loaded {data['target_name']}")
    print(f"✓ Receptor: {len(list(data['receptor_unbound'].get_atoms()))} atoms")
    print(f"✓ Ligand: {len(list(data['ligand_unbound'].get_atoms()))} atoms")
