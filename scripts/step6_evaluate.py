#!/usr/bin/env python3
"""
Step 6: Evaluation — CAPRI Metrics (DockQ, fnat, I-RMSD, L-RMSD)
Evaluate trained model on all BM5.5 targets and compare with AlphaRED.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
from tqdm import tqdm
from scipy.spatial.distance import cdist

from step2_data_loader import ProteinDockingLoader
from step3_feature_extraction import ProteinFeatureExtractor
from step4_model import ProteinDockingModel, prepare_graph_data


# ============================================================
# CAPRI metric functions
# ============================================================

def compute_fnat(pred_rec_coords, pred_lig_coords,
                 native_rec_coords, native_lig_coords, threshold=5.0):
    """
    Fraction of native contacts preserved in prediction.
    Native contacts: receptor-ligand Cα pairs within threshold in bound complex.
    """
    native_dists = cdist(native_rec_coords, native_lig_coords)
    native_contacts = set(zip(*np.where(native_dists < threshold)))

    if len(native_contacts) == 0:
        return 0.0

    pred_dists = cdist(pred_rec_coords, pred_lig_coords)
    pred_contacts = set(zip(*np.where(pred_dists < threshold)))

    return len(native_contacts & pred_contacts) / len(native_contacts)


def kabsch_superimpose(mobile, target):
    """
    Kabsch algorithm: find optimal rotation+translation to align mobile onto target.
    Returns: R (3x3), t (3,) such that mobile @ R.T + t ≈ target
    """
    n = min(len(mobile), len(target))
    mobile, target = mobile[:n], target[:n]

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    t = target_center - mobile_center @ R.T
    return R, t


def compute_lrmsd(pred_rec, pred_lig, native_rec, native_lig):
    """
    Ligand-RMSD: superimpose on receptor (Kabsch), measure ligand deviation.
    """
    n_rec = min(len(pred_rec), len(native_rec))
    n_lig = min(len(pred_lig), len(native_lig))

    R, t = kabsch_superimpose(pred_rec[:n_rec], native_rec[:n_rec])
    aligned_lig = pred_lig[:n_lig] @ R.T + t

    diff = aligned_lig - native_lig[:n_lig]
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_irmsd(pred_rec, pred_lig, native_rec, native_lig,
                  interface_cutoff=10.0):
    """
    Interface-RMSD: RMSD of interface residues after superimposing on interface.
    """
    native_dists = cdist(native_rec, native_lig)
    rec_iface_idx = np.where(native_dists.min(axis=1) < interface_cutoff)[0]
    lig_iface_idx = np.where(native_dists.min(axis=0) < interface_cutoff)[0]

    if len(rec_iface_idx) == 0 or len(lig_iface_idx) == 0:
        # Fallback to simple RMSD of ligand
        n = min(len(pred_lig), len(native_lig))
        diff = pred_lig[:n] - native_lig[:n]
        return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    # Collect interface atoms (native and predicted)
    native_iface = np.vstack([native_rec[rec_iface_idx], native_lig[lig_iface_idx]])
    pred_iface = np.vstack([pred_rec[rec_iface_idx], pred_lig[lig_iface_idx]])

    # Superimpose on interface, compute RMSD
    R, t = kabsch_superimpose(pred_iface, native_iface)
    aligned = pred_iface @ R.T + t

    diff = aligned - native_iface
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_dockq(irmsd, lrmsd, fnat):
    """DockQ = (fnat + 1/(1+(irmsd/1.5)²) + 1/(1+(lrmsd/8.5)²)) / 3"""
    return (fnat
            + 1.0 / (1.0 + (irmsd / 1.5) ** 2)
            + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)) / 3.0


def capri_quality(dockq):
    if dockq >= 0.8:
        return 'high'
    elif dockq >= 0.49:
        return 'medium'
    elif dockq >= 0.23:
        return 'acceptable'
    return 'incorrect'


# ============================================================
# Evaluation logic
# ============================================================

def evaluate_single(model, features, device):
    """Evaluate one target, return full CAPRI metrics."""
    receptor_data, ligand_data, bound_coords = prepare_graph_data(features, device)

    with torch.no_grad():
        rotation, translation, confidence = model(receptor_data, ligand_data)
        pred_lig = model.apply_transformation(
            ligand_data['coords'], rotation, translation
        )

    pred_rec_np = receptor_data['coords'].cpu().numpy()
    pred_lig_np = pred_lig.cpu().numpy()
    native_rec_np = features['receptor']['coords']
    native_lig_np = bound_coords.cpu().numpy()  # bound ligand = ground truth

    fnat = compute_fnat(pred_rec_np, pred_lig_np, native_rec_np, native_lig_np)
    lrmsd = compute_lrmsd(pred_rec_np, pred_lig_np, native_rec_np, native_lig_np)
    irmsd = compute_irmsd(pred_rec_np, pred_lig_np, native_rec_np, native_lig_np)
    dockq = compute_dockq(irmsd, lrmsd, fnat)

    return {
        'irmsd': float(irmsd),
        'lrmsd': float(lrmsd),
        'fnat': float(fnat),
        'dockq': float(dockq),
        'quality': capri_quality(dockq),
        'success': dockq >= 0.23,
        'confidence': float(confidence.item()),
        'error': None,
    }


def evaluate_all(model, device):
    """Run evaluation on all BM5.5 targets."""
    loader = ProteinDockingLoader()
    extractor = ProteinFeatureExtractor()

    difficulties = ["rigid_targets", "medium_targets", "difficult_targets"]
    rows = []

    for difficulty in difficulties:
        bench = Path(loader.benchmark_path) / difficulty
        targets = sorted([d for d in bench.iterdir() if d.is_dir()])
        print(f"\n{difficulty} ({len(targets)} targets)")

        for tdir in tqdm(targets, desc=f"  {difficulty}"):
            name = tdir.name
            try:
                features = extractor.extract_full_features(name, difficulty)
                result = evaluate_single(model, features, device)
            except Exception as e:
                result = dict(irmsd=None, lrmsd=None, fnat=None, dockq=None,
                              quality=None, success=False, confidence=None,
                              error=str(e))
            rows.append({'target': name, 'difficulty': difficulty, **result})

    return pd.DataFrame(rows)


def print_results(df):
    """Print formatted results summary."""
    valid = df[df['error'].isna()]
    total = len(valid)
    if total == 0:
        print("No valid results.")
        return

    # Overall
    print(f"\n{'=' * 80}")
    print("OVERALL PERFORMANCE")
    print(f"{'=' * 80}")
    successes = int(valid['success'].sum())
    print(f"  Targets evaluated: {total}")
    print(f"  Success (DockQ >= 0.23): {successes} ({successes / total * 100:.1f}%)")

    for label, thresh in [('high', 0.8), ('medium', 0.49), ('acceptable', 0.23)]:
        n = int((valid['dockq'] >= thresh).sum())
        print(f"  {label:>12} (DockQ >= {thresh}): {n} ({n / total * 100:.1f}%)")

    print(f"\n  Mean DockQ:  {valid['dockq'].mean():.4f} +/- {valid['dockq'].std():.4f}")
    print(f"  Mean I-RMSD: {valid['irmsd'].mean():.4f} +/- {valid['irmsd'].std():.4f} A")
    print(f"  Mean L-RMSD: {valid['lrmsd'].mean():.4f} +/- {valid['lrmsd'].std():.4f} A")
    print(f"  Mean fnat:   {valid['fnat'].mean():.4f} +/- {valid['fnat'].std():.4f}")

    # By difficulty
    print(f"\n{'=' * 80}")
    print("BY DIFFICULTY")
    print(f"{'=' * 80}")
    for diff in ["rigid_targets", "medium_targets", "difficult_targets"]:
        sub = valid[valid['difficulty'] == diff]
        if len(sub) == 0:
            continue
        s = int(sub['success'].sum())
        print(f"\n  {diff} ({len(sub)} targets):")
        print(f"    Success: {s} ({s / len(sub) * 100:.1f}%)")
        print(f"    Mean DockQ:  {sub['dockq'].mean():.4f}")
        print(f"    Mean I-RMSD: {sub['irmsd'].mean():.4f} A")
        print(f"    Mean fnat:   {sub['fnat'].mean():.4f}")

    # Comparison with AlphaRED
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH ALPHARED")
    print(f"{'=' * 80}")
    our_rate = successes / total * 100
    print(f"  AlphaRED:   63.0% success (DockQ >= 0.23)")
    print(f"  Our model:  {our_rate:.1f}% success")
    print(f"  Difference: {our_rate - 63.0:+.1f} percentage points")

    errors = df[df['error'].notna()]
    if len(errors) > 0:
        print(f"\n  Targets with errors: {len(errors)}")
        for _, row in errors.head(5).iterrows():
            print(f"    {row['target']}: {row['error'][:60]}")


# ============================================================
# Main
# ============================================================

def find_best_checkpoint():
    """Auto-detect most recent best_model.pt."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    experiments = project_root / 'experiments'
    if not experiments.exists():
        experiments = script_dir / 'experiments'
    candidates = sorted(experiments.glob('*/best_model.pt'))
    if candidates:
        return candidates[-1]  # most recent
    return None


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find checkpoint
    ckpt = find_best_checkpoint()
    if len(sys.argv) > 1:
        ckpt = Path(sys.argv[1])

    if ckpt is None or not ckpt.exists():
        print(f"No checkpoint found. Usage: python step6_evaluate.py [path/to/best_model.pt]")
        return

    print("=" * 80)
    print("CAPRI EVALUATION (DockQ, fnat, I-RMSD, L-RMSD)")
    print("=" * 80)
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device: {device}")

    # Load model
    model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print(f"  Loaded epoch {checkpoint['epoch']}, "
          f"val RMSD {checkpoint.get('val_loss', '?')}")

    # Evaluate
    df = evaluate_all(model, device)

    # Print
    print_results(df)

    # Save
    out_dir = ckpt.parent / 'evaluation'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'capri_results.csv', index=False)

    valid = df[df['error'].isna()]
    summary = {
        'checkpoint': str(ckpt),
        'total': len(valid),
        'success_count': int(valid['success'].sum()),
        'success_rate': float(valid['success'].mean() * 100),
        'mean_dockq': float(valid['dockq'].mean()),
        'mean_irmsd': float(valid['irmsd'].mean()),
        'mean_lrmsd': float(valid['lrmsd'].mean()),
        'mean_fnat': float(valid['fnat'].mean()),
    }
    for diff in ["rigid_targets", "medium_targets", "difficult_targets"]:
        sub = valid[valid['difficulty'] == diff]
        if len(sub) > 0:
            summary[diff] = {
                'count': len(sub),
                'success_rate': float(sub['success'].mean() * 100),
                'mean_dockq': float(sub['dockq'].mean()),
            }

    with open(out_dir / 'capri_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()