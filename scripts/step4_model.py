#!/usr/bin/env python3
"""
Step 4: Graph Neural Network for Protein Docking
Dual-encoder GNN that predicts rigid-body transformation (rotation + translation)
to align unbound ligand to bound position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

from step3_feature_extraction import ProteinFeatureExtractor


class ProteinGraphConv(nn.Module):
    """Graph convolution with scatter-based message passing"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, in_features]
            edge_index: [2, E] (src, dst)
        Returns:
            [N, out_features]
        """
        x_transformed = self.fc(x)

        # Scatter-add message passing (vectorized, no Python loop)
        src, dst = edge_index[0], edge_index[1]
        messages = x_transformed[src]  # [E, out_features]
        x_aggregated = x_transformed.clone()
        x_aggregated.index_add_(0, dst, messages)

        x_out = self.norm(x_aggregated)
        x_out = F.relu(x_out)
        return x_out


class ProteinEncoder(nn.Module):
    """Encode protein graph into fixed-size embedding"""

    def __init__(self, node_features=26, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(node_features, hidden_dim)
        self.conv_layers = nn.ModuleList([
            ProteinGraphConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.pool = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [N, 26]
            edge_index: [2, E]
        Returns:
            node_embeddings: [N, hidden_dim]
            graph_embedding: [hidden_dim]
        """
        x = F.relu(self.input_proj(node_features))

        for conv in self.conv_layers:
            x = conv(x, edge_index) + x  # residual

        graph_embedding = self.pool(torch.mean(x, dim=0))
        return x, graph_embedding


class DockingHead(nn.Module):
    """Predict rotation (axis-angle), translation, and confidence"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rotation_head = nn.Linear(hidden_dim, 3)
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, receptor_embedding, ligand_embedding):
        combined = torch.cat([receptor_embedding, ligand_embedding], dim=0)
        features = self.combine(combined)
        rotation = self.rotation_head(features)
        translation = self.translation_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))
        return rotation, translation, confidence


class ProteinDockingModel(nn.Module):
    """Complete dual-encoder protein docking model"""

    def __init__(self, node_features=26, hidden_dim=128, num_layers=3):
        super().__init__()
        self.receptor_encoder = ProteinEncoder(node_features, hidden_dim, num_layers)
        self.ligand_encoder = ProteinEncoder(node_features, hidden_dim, num_layers)
        self.docking_head = DockingHead(hidden_dim)

    def forward(self, receptor_data, ligand_data):
        _, rec_emb = self.receptor_encoder(
            receptor_data['node_features'], receptor_data['edge_index']
        )
        _, lig_emb = self.ligand_encoder(
            ligand_data['node_features'], ligand_data['edge_index']
        )
        rotation, translation, confidence = self.docking_head(rec_emb, lig_emb)
        return rotation, translation, confidence

    def apply_transformation(self, coords, rotation, translation):
        """
        Apply rigid-body transformation using Rodrigues' formula.

        Args:
            coords: [N, 3]
            rotation: [3] axis-angle vector
            translation: [3]
        Returns:
            [N, 3] transformed coordinates
        """
        angle = torch.norm(rotation)

        if angle < 1e-6:
            R = torch.eye(3, device=coords.device)
        else:
            axis = rotation / angle
            K = torch.zeros(3, 3, device=coords.device)
            K[0, 1] = -axis[2]
            K[0, 2] = axis[1]
            K[1, 0] = axis[2]
            K[1, 2] = -axis[0]
            K[2, 0] = -axis[1]
            K[2, 1] = axis[0]
            # R = I + sin(θ)K + (1 - cos(θ))K²
            R = (torch.eye(3, device=coords.device)
                 + torch.sin(angle) * K
                 + (1 - torch.cos(angle)) * K @ K)

        return coords @ R.t() + translation.unsqueeze(0)


def prepare_graph_data(features, device='cpu'):
    """
    Convert step3 output to PyTorch tensors for the model.

    Args:
        features: dict from ProteinFeatureExtractor.extract_full_features()
        device: torch device
    Returns:
        receptor_data, ligand_data, bound_coords
    """
    rec = features['receptor']
    lig = features['ligand']

    receptor_data = {
        'node_features': torch.tensor(rec['node_features'], dtype=torch.float32).to(device),
        'edge_index': torch.tensor(rec['edges'], dtype=torch.long).to(device),
        'coords': torch.tensor(rec['coords'], dtype=torch.float32).to(device),
    }

    ligand_data = {
        'node_features': torch.tensor(lig['node_features'], dtype=torch.float32).to(device),
        'edge_index': torch.tensor(lig['edges'], dtype=torch.long).to(device),
        'coords': torch.tensor(lig['coords'], dtype=torch.float32).to(device),
    }

    bound_coords = torch.tensor(
        features['bound_ligand_coords'], dtype=torch.float32
    ).to(device)

    return receptor_data, ligand_data, bound_coords


def test_model(target_name="1A2K", difficulty="rigid_targets"):
    """Test model on one target"""
    print("=" * 80)
    print(f"TESTING MODEL: {target_name}")
    print("=" * 80)

    # 1. Extract features
    print("\n1. Extracting features...")
    extractor = ProteinFeatureExtractor()
    features = extractor.extract_full_features(target_name, difficulty)
    print("   Done")

    # 2. Prepare tensors
    print("2. Preparing graph data...")
    receptor_data, ligand_data, bound_coords = prepare_graph_data(features)
    print(f"   Receptor: {receptor_data['node_features'].shape}, "
          f"{receptor_data['edge_index'].shape[1]} edges")
    print(f"   Ligand:   {ligand_data['node_features'].shape}, "
          f"{ligand_data['edge_index'].shape[1]} edges")
    print(f"   Bound:    {bound_coords.shape}")

    # 3. Create model
    print("3. Creating model...")
    model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # 4. Forward pass
    print("4. Forward pass...")
    with torch.no_grad():
        rotation, translation, confidence = model(receptor_data, ligand_data)
    print(f"   Rotation:    {rotation.numpy().round(6)}")
    print(f"   Translation: {translation.numpy().round(4)}")
    print(f"   Confidence:  {confidence.item():.4f}")

    # 5. Apply transformation and evaluate
    print("5. Applying transformation...")
    with torch.no_grad():
        predicted_coords = model.apply_transformation(
            ligand_data['coords'], rotation, translation
        )
    rmsd = torch.sqrt(torch.mean(torch.sum(
        (predicted_coords - bound_coords) ** 2, dim=1
    ))).item()
    print(f"   RMSD to ground truth: {rmsd:.4f} A (untrained model)")

    print(f"\n{'=' * 80}")
    print("MODEL TEST PASSED")
    print(f"{'=' * 80}")
    return model


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "1A2K"
    diff = sys.argv[2] if len(sys.argv) > 2 else "rigid_targets"
    test_model(target, diff)