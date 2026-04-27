#!/usr/bin/env python3
"""
Compute TM-score between DockGraph predicted complex and ground truth bound complex.
TM-score formula: Zhang & Skolnick, Proteins 57:702-710 (2004)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from step2_data_loader import ProteinDockingLoader
from step3_feature_extraction import ProteinFeatureExtractor
from step4_model import ProteinDockingModel, prepare_graph_data


def kabsch_superpose(P, Q):
    """Superpose P onto Q using Kabsch algorithm. Returns rotated P."""
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return P_c @ R.T + centroid_Q


def compute_tm_score(coords_pred, coords_native):
    """
    TM-score = (1/L) * Σ [1 / (1 + (d_i / d_0)^2)]
    d_0 = 1.24 * (L - 15)^(1/3) - 1.8
    """
    L = len(coords_native)
    if L <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8
        d0 = max(d0, 0.5)

    coords_aligned = kabsch_superpose(coords_pred, coords_native)
    distances = np.sqrt(np.sum((coords_aligned - coords_native) ** 2, axis=1))
    tm = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L
    return tm


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    candidates = sorted(project_root.glob('experiments/*/best_model.pt'))
    if not candidates:
        print("No checkpoint found.")
        return
    ckpt = candidates[-1]

    print("=" * 70)
    print("TM-SCORE COMPUTATION")
    print("=" * 70)
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device: {device}")

    model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print(f"  Model loaded from epoch {checkpoint['epoch']}")

    loader = ProteinDockingLoader()
    extractor = ProteinFeatureExtractor()

    eval_dir = ckpt.parent / 'evaluation'
    capri = pd.read_csv(eval_dir / 'capri_results.csv')

    rows = []

    for _, row in tqdm(capri.iterrows(), total=len(capri), desc="  Computing TM-scores"):
        target = row['target']
        difficulty = row['difficulty']
        dockq = row.get('dockq', None)
        success = row.get('success', False)

        if pd.notna(row.get('error')):
            rows.append({
                'target': target, 'difficulty': difficulty,
                'dockq': dockq, 'success': success,
                'tm_score_complex': None, 'tm_score_ligand': None,
                'n_residues_complex': None, 'error': row['error']
            })
            continue

        try:
            features = extractor.extract_full_features(target, difficulty)

            # Extract data matching actual dictionary structure
            rec_coords = features['receptor']['coords']          # (N, 3)
            lig_coords = features['ligand']['coords']            # (M, 3)
            rec_features_arr = features['receptor']['node_features']  # (N, 26)
            lig_features_arr = features['ligand']['node_features']    # (M, 26)
            rec_edges = features['receptor']['edges']            # (2, E_r)
            lig_edges = features['ligand']['edges']              # (2, E_l)
            bound_lig_coords = features['bound_ligand_coords']   # (M, 3)

            # Prepare graph data and run model
            graph_data = prepare_graph_data(
                rec_features_arr, lig_features_arr,
                rec_edges, lig_edges
            )

            with torch.no_grad():
                graph_data_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in graph_data.items()}
                output = model(graph_data_device)

            rotation = output['rotation'].cpu()
            translation = output['translation'].cpu()

            # Apply transformation to ligand
            lig_tensor = torch.tensor(lig_coords, dtype=torch.float32)
            pred_lig_coords = model.apply_transformation(lig_tensor, rotation, translation).detach().cpu().numpy()

            # TM-score on full complex (receptor + ligand)
            pred_complex = np.concatenate([rec_coords, pred_lig_coords], axis=0)
            native_complex = np.concatenate([rec_coords, bound_lig_coords], axis=0)

            tm_complex = compute_tm_score(pred_complex, native_complex)

            # TM-score on ligand only
            tm_ligand = compute_tm_score(pred_lig_coords, bound_lig_coords)

            rows.append({
                'target': target, 'difficulty': difficulty,
                'dockq': dockq, 'success': success,
                'tm_score_complex': round(tm_complex, 4),
                'tm_score_ligand': round(tm_ligand, 4),
                'n_residues_complex': len(native_complex),
                'error': None
            })

        except Exception as e:
            rows.append({
                'target': target, 'difficulty': difficulty,
                'dockq': dockq, 'success': success,
                'tm_score_complex': None, 'tm_score_ligand': None,
                'n_residues_complex': None, 'error': str(e)
            })

    df = pd.DataFrame(rows)
    valid = df[df['error'].isna()]

    print(f"\n{'=' * 70}")
    print("TM-SCORE RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total targets: {len(df)}")
    print(f"  Successfully computed: {len(valid)}")
    if df['error'].notna().sum() > 0:
        print(f"  Errors: {df['error'].notna().sum()}")

    print(f"\n  TM-score (complex):")
    print(f"    Mean:   {valid['tm_score_complex'].mean():.4f} +/- {valid['tm_score_complex'].std():.4f}")
    print(f"    Median: {valid['tm_score_complex'].median():.4f}")
    print(f"    Min:    {valid['tm_score_complex'].min():.4f}")
    print(f"    Max:    {valid['tm_score_complex'].max():.4f}")
    print(f"    TM >= 0.5 (same fold): {(valid['tm_score_complex'] >= 0.5).sum()}/{len(valid)} "
          f"({(valid['tm_score_complex'] >= 0.5).mean()*100:.1f}%)")
    print(f"    TM >= 0.8 (high sim.): {(valid['tm_score_complex'] >= 0.8).sum()}/{len(valid)} "
          f"({(valid['tm_score_complex'] >= 0.8).mean()*100:.1f}%)")

    print(f"\n  TM-score (ligand only):")
    print(f"    Mean:   {valid['tm_score_ligand'].mean():.4f} +/- {valid['tm_score_ligand'].std():.4f}")
    print(f"    Median: {valid['tm_score_ligand'].median():.4f}")

    print(f"\n  BY DIFFICULTY:")
    for diff in ['rigid_targets', 'medium_targets', 'difficult_targets']:
        sub = valid[valid['difficulty'] == diff]
        if len(sub) == 0:
            continue
        print(f"    {diff:<18}: TM(complex)={sub['tm_score_complex'].mean():.4f}  "
              f"TM(ligand)={sub['tm_score_ligand'].mean():.4f}  n={len(sub)}")

    succ = valid[valid['success'] == True]
    fail = valid[valid['success'] == False]
    print(f"\n  SUCCESS vs FAILURE:")
    if len(succ) > 0:
        print(f"    Successful (n={len(succ)}): TM(complex)={succ['tm_score_complex'].mean():.4f}")
    if len(fail) > 0:
        print(f"    Failed     (n={len(fail)}): TM(complex)={fail['tm_score_complex'].mean():.4f}")

    corr = valid[['dockq', 'tm_score_complex', 'tm_score_ligand']].corr()
    print(f"\n  CORRELATION WITH DockQ:")
    print(f"    TM(complex) vs DockQ: r = {corr.loc['dockq', 'tm_score_complex']:.4f}")
    print(f"    TM(ligand)  vs DockQ: r = {corr.loc['dockq', 'tm_score_ligand']:.4f}")

    out_path = eval_dir / 'tm_scores.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()