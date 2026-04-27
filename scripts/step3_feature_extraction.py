#!/usr/bin/env python3
"""
Step 3: Feature Extraction for Deep Learning
Extract geometric and chemical features from proteins
26D node features: 3 (coords) + 20 (AA one-hot) + 3 (biophysical properties)
8 Angstrom contact graph for edges
"""

import numpy as np
from Bio import PDB
from scipy.spatial.distance import cdist
from pathlib import Path
import sys

# Fix import
from step2_data_loader import ProteinDockingLoader

# Amino acid properties: [hydrophobic, positive_charge, negative_charge]
AA_PROPERTIES = {
    'ALA': [0, 0, 0], 'ARG': [0, 1, 0], 'ASN': [0, 0, 0], 'ASP': [0, 0, 1],
    'CYS': [0, 0, 0], 'GLN': [0, 0, 0], 'GLU': [0, 0, 1], 'GLY': [0, 0, 0],
    'HIS': [0, 1, 0], 'ILE': [1, 0, 0], 'LEU': [1, 0, 0], 'LYS': [0, 1, 0],
    'MET': [1, 0, 0], 'PHE': [1, 0, 0], 'PRO': [0, 0, 0], 'SER': [0, 0, 0],
    'THR': [0, 0, 0], 'TRP': [1, 0, 0], 'TYR': [1, 0, 0], 'VAL': [1, 0, 0],
}

AA_LIST = sorted(AA_PROPERTIES.keys())  # Alphabetical: ALA, ARG, ASN, ...
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}


class ProteinFeatureExtractor:
    """Extract 26D node features and 8A contact graphs from protein structures"""

    def __init__(self, benchmark_path=None):
        self.loader = ProteinDockingLoader(benchmark_path)

    def extract_residue_features(self, structure, chain_filter=None):
        """
        Extract per-residue features from a structure.

        Args:
            structure: Bio.PDB Structure
            chain_filter: list of chain IDs to include (None = all chains)

        Returns:
            dict with coords (Nx3), node_features (Nx26), num_residues (int)
        """
        coords = []
        one_hot = []
        properties = []

        model = next(structure.get_models())
        for chain in model:
            if chain_filter is not None and chain.id not in chain_filter:
                continue
            for residue in chain:
                if residue.id[0] != ' ':  # skip HETATMs
                    continue
                if 'CA' not in residue:
                    continue

                ca_coord = residue['CA'].get_coord()
                res_name = residue.resname

                coords.append(ca_coord)

                # 20D one-hot encoding
                oh = [0.0] * 20
                if res_name in AA_TO_INDEX:
                    oh[AA_TO_INDEX[res_name]] = 1.0
                one_hot.append(oh)

                # 3D biophysical properties
                props = AA_PROPERTIES.get(res_name, [0, 0, 0])
                properties.append(props)

        coords = np.array(coords, dtype=np.float32)
        one_hot = np.array(one_hot, dtype=np.float32)
        properties = np.array(properties, dtype=np.float32)

        # 26D node features: [x, y, z, one_hot(20), properties(3)]
        node_features = np.concatenate([coords, one_hot, properties], axis=1)

        return {
            'coords': coords,
            'node_features': node_features,
            'num_residues': len(coords),
        }

    def build_contact_graph(self, coords, threshold=8.0):
        """
        Build contact graph using distance threshold.

        Args:
            coords: Nx3 array of CA positions
            threshold: Angstrom cutoff (default 8.0)

        Returns:
            edges: (2, E) array of edge indices
            edge_distances: (E,) array of distances
        """
        dist_matrix = cdist(coords, coords)
        i_idx, j_idx = np.where((dist_matrix < threshold) & (dist_matrix > 0))

        # Keep only upper triangle (undirected graph stored both ways for message passing)
        edges = np.stack([i_idx, j_idx], axis=0)  # shape (2, E)
        edge_distances = dist_matrix[i_idx, j_idx]

        return edges, edge_distances

    def _parse_partners(self, target_dir, target_name):
        """Parse partners file to get receptor/ligand chain IDs."""
        partners_file = target_dir / "partners"
        if not partners_file.exists():
            return None, None

        content = open(partners_file).read().strip()
        # Format: '-partners AB_C' or 'AB_C'
        content = content.replace('-partners', '').strip()
        for sep in ['_', ':']:
            if sep in content:
                parts = content.split(sep)
                return list(parts[0]), list(parts[1])
        return list(content[0]), list(content[1:])

    def extract_bound_ligand_coords(self, bound_structure, ligand_chains):
        """
        Extract ligand CA coordinates from bound complex using chain IDs.

        Args:
            bound_structure: Bio.PDB Structure of the bound complex
            ligand_chains: list of chain IDs belonging to the ligand

        Returns:
            np.array of shape (M, 3)
        """
        coords = []
        model = next(bound_structure.get_models())
        for chain in model:
            if chain.id in ligand_chains:
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
        return np.array(coords, dtype=np.float32)

    def extract_full_features(self, target_name, difficulty="rigid_targets"):
        """
        Extract all features for a docking target.

        Returns:
            dict with receptor features, ligand features, bound coords, edges
        """
        # Resolve paths
        target_dir = self.loader.benchmark_path / difficulty / target_name
        rec_chains, lig_chains = self._parse_partners(target_dir, target_name)

        # Load structures
        data = self.loader.load_target(target_name, difficulty)

        # Receptor features (unbound) — filter to receptor chains only
        receptor = self.extract_residue_features(
            data['receptor_unbound'], chain_filter=rec_chains
        )
        rec_edges, rec_edge_dist = self.build_contact_graph(receptor['coords'])
        receptor['edges'] = rec_edges
        receptor['edge_distances'] = rec_edge_dist

        # Ligand features (unbound) — filter to ligand chains only
        ligand = self.extract_residue_features(
            data['ligand_unbound'], chain_filter=lig_chains
        )
        lig_edges, lig_edge_dist = self.build_contact_graph(ligand['coords'])
        ligand['edges'] = lig_edges
        ligand['edge_distances'] = lig_edge_dist

        # Ground truth: ligand coordinates in the bound complex
        bound_ligand_coords = self.extract_bound_ligand_coords(
            data['bound_complex'], lig_chains
        )

        return {
            'target_name': target_name,
            'difficulty': difficulty,
            'receptor_chains': rec_chains,
            'ligand_chains': lig_chains,
            'receptor': receptor,
            'ligand': ligand,
            'bound_ligand_coords': bound_ligand_coords,
        }


def test_feature_extraction(target_name="1A2K", difficulty="rigid_targets"):
    """Test feature extraction on one target"""
    print("=" * 80)
    print(f"FEATURE EXTRACTION: {target_name} ({difficulty})")
    print("=" * 80)

    extractor = ProteinFeatureExtractor()
    features = extractor.extract_full_features(target_name, difficulty)

    rec = features['receptor']
    lig = features['ligand']
    bound = features['bound_ligand_coords']

    print(f"\n  Receptor chains: {features['receptor_chains']}")
    print(f"  Ligand chains:   {features['ligand_chains']}")

    print(f"\n  Receptor: {rec['num_residues']} residues, "
          f"node_features {rec['node_features'].shape}, "
          f"{rec['edges'].shape[1]} edges")

    print(f"  Ligand:   {lig['num_residues']} residues, "
          f"node_features {lig['node_features'].shape}, "
          f"{lig['edges'].shape[1]} edges")

    print(f"  Bound ligand: {bound.shape}")

    # Sanity check: bound ligand residue count should match unbound ligand
    if bound.shape[0] != lig['num_residues']:
        print(f"  WARNING: bound ({bound.shape[0]}) != unbound ({lig['num_residues']}) residue count")
    else:
        print(f"  Residue count match: OK")

    # Distances
    rec_center = rec['coords'].mean(axis=0)
    lig_center = lig['coords'].mean(axis=0)
    bound_center = bound.mean(axis=0)

    print(f"\n  Receptor center:      [{rec_center[0]:.1f}, {rec_center[1]:.1f}, {rec_center[2]:.1f}]")
    print(f"  Ligand unbound center: [{lig_center[0]:.1f}, {lig_center[1]:.1f}, {lig_center[2]:.1f}]")
    print(f"  Ligand bound center:   [{bound_center[0]:.1f}, {bound_center[1]:.1f}, {bound_center[2]:.1f}]")

    move_needed = np.linalg.norm(bound_center - lig_center)
    print(f"\n  Ligand displacement needed: {move_needed:.2f} A")
    print(f"  Task: {'refinement' if move_needed < 5.0 else 'global docking'}")

    print(f"\n  Feature vector breakdown (26D):")
    print(f"    [0:3]   = CA coordinates (3D)")
    print(f"    [3:23]  = amino acid one-hot (20D)")
    print(f"    [23:26] = biophysical properties (3D)")

    return features


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "1A2K"
    diff = sys.argv[2] if len(sys.argv) > 2 else "rigid_targets"
    test_feature_extraction(target, diff)