#!/usr/bin/env python3
"""
Batch Prediction + TM-Score for DB5.5 Benchmark
=================================================
Predicts docking for all targets, saves predicted PDBs, computes TM-scores.

Usage (TorchScript model):
    python batch_predict.py --benchmark data/benchmark --model experiments/pretrained/dockgraph.pt

Usage (regular checkpoint):
    python batch_predict.py --benchmark data/benchmark --checkpoint experiments/20260307_073713_full_training/best_model.pt
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
from Bio.PDB import PDBParser, PDBIO, Structure, Model
from scipy.spatial.distance import cdist


# ============================================================
# Feature Extraction
# ============================================================
_AA_PROPS = {
    'ALA': [0,0,0], 'ARG': [0,1,0], 'ASN': [0,0,0], 'ASP': [0,0,1],
    'CYS': [0,0,0], 'GLN': [0,0,0], 'GLU': [0,0,1], 'GLY': [0,0,0],
    'HIS': [0,1,0], 'ILE': [1,0,0], 'LEU': [1,0,0], 'LYS': [0,1,0],
    'MET': [1,0,0], 'PHE': [1,0,0], 'PRO': [0,0,0], 'SER': [0,0,0],
    'THR': [0,0,0], 'TRP': [1,0,0], 'TYR': [1,0,0], 'VAL': [1,0,0],
}
_AA_IDX = {aa: i for i, aa in enumerate(sorted(_AA_PROPS))}


def extract_features(pdb_path, chains=None):
    """Extract 26D features + contact graph from PDB."""
    s = PDBParser(QUIET=True).get_structure('p', pdb_path)
    coords, oh, props = [], [], []
    for ch in next(s.get_models()):
        if chains and ch.id not in chains:
            continue
        for r in ch:
            if r.id[0] != ' ' or 'CA' not in r:
                continue
            coords.append(r['CA'].get_coord())
            o = [0.0] * 20
            if r.resname in _AA_IDX:
                o[_AA_IDX[r.resname]] = 1.0
            oh.append(o)
            props.append(_AA_PROPS.get(r.resname, [0, 0, 0]))
    c = np.array(coords, dtype=np.float32)
    nf = np.concatenate([c, np.array(oh, np.float32), np.array(props, np.float32)], 1)
    dm = cdist(c, c)
    i, j = np.where((dm < 8.0) & (dm > 0))
    return {'coords': c, 'node_features': nf, 'edges': np.stack([i, j])}


def get_bound_ligand_coords(bound_pdb, lig_chains):
    """Extract Cα coords of ligand chains from bound complex."""
    s = PDBParser(QUIET=True).get_structure('b', bound_pdb)
    coords = []
    for ch in next(s.get_models()):
        if ch.id not in lig_chains:
            continue
        for r in ch:
            if r.id[0] != ' ' or 'CA' not in r:
                continue
            coords.append(r['CA'].get_coord())
    return np.array(coords, dtype=np.float32)


def parse_partners(partners_file):
    """Parse partners file to get chain IDs."""
    if not partners_file.exists():
        return None, None
    with open(partners_file) as f:
        line = f.readline().strip()
        if '|' not in line:
            return None, None
        parts = line.split('|')
        rec_chains = [c.strip() for c in parts[0].split(',') if c.strip()]
        lig_chains = [c.strip() for c in parts[1].split(',') if c.strip()]
        return rec_chains, lig_chains


def get_chains(pdb):
    """Get chain IDs from PDB."""
    s = PDBParser(QUIET=True).get_structure('s', pdb)
    return [c.id for c in next(s.get_models()).get_chains()]


# ============================================================
# Transformation
# ============================================================
def apply_rodrigues(coords, rotation, translation):
    """Apply Rodrigues rotation + translation to coordinates."""
    angle = torch.norm(rotation)
    if angle < 1e-6:
        R = torch.eye(3, device=coords.device)
    else:
        ax = rotation / angle
        K = torch.zeros(3, 3, device=coords.device)
        K[0, 1], K[0, 2] = -ax[2], ax[1]
        K[1, 0], K[1, 2] = ax[2], -ax[0]
        K[2, 0], K[2, 1] = -ax[1], ax[0]
        R = torch.eye(3, device=coords.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    return coords @ R.t() + translation.unsqueeze(0)


# ============================================================
# TM-Score
# ============================================================
def kabsch_superpose(P, Q):
    """Superpose P onto Q using Kabsch algorithm."""
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Pc @ R.T + cQ


def compute_tm_score(pred, native):
    """TM-score (Zhang & Skolnick 2004)."""
    L = len(native)
    d0 = max(1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8, 0.5) if L > 15 else 0.5
    aligned = kabsch_superpose(pred, native)
    d = np.sqrt(np.sum((aligned - native) ** 2, axis=1))
    return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / L)


# ============================================================
# PDB Writing
# ============================================================
def write_predicted_pdb(rec_pdb, lig_pdb, pred_coords, rc, lc, out):
    """Write predicted complex PDB."""
    parser = PDBParser(QUIET=True)
    ns = Structure.Structure('pred')
    nm = Model.Model(0)
    ns.add(nm)

    for ch in next(parser.get_structure('r', rec_pdb).get_models()):
        if rc is None or ch.id in rc:
            nm.add(ch.copy())

    ci = 0
    for ch in next(parser.get_structure('l', lig_pdb).get_models()):
        if lc and ch.id not in lc:
            continue
        nc = ch.copy()
        for r in nc:
            if r.id[0] != ' ' or 'CA' not in r:
                continue
            if ci >= len(pred_coords):
                break
            d = pred_coords[ci] - r['CA'].get_vector().get_array()
            for a in r:
                a.set_coord(a.get_coord() + d)
            ci += 1
        if nm.has_id(nc.id):
            for x in 'XYZWVU':
                if not nm.has_id(x):
                    nc.id = x
                    break
        nm.add(nc)

    io = PDBIO()
    io.set_structure(ns)
    io.save(str(out))


# ============================================================
# Target Discovery
# ============================================================
def find_all_targets(benchmark_dir):
    """Find all targets in DB5.5 benchmark."""
    benchmark_path = Path(benchmark_dir)
    targets = {}

    for difficulty in ['rigid', 'medium', 'difficult']:
        diff_dir = benchmark_path / f'{difficulty}_targets'
        if not diff_dir.exists():
            continue

        targets[difficulty] = []
        for target_dir in sorted(diff_dir.iterdir()):
            if not target_dir.is_dir():
                continue

            tid = target_dir.name
            rec_u = target_dir / f'{tid}_r_u.pdb'
            lig_u = target_dir / f'{tid}_l_u.pdb'
            bound = target_dir / f'{tid}_b.pdb'
            lig_b = target_dir / f'{tid}_l_b.pdb'
            partners = target_dir / 'partners'

            if rec_u.exists() and lig_u.exists():
                targets[difficulty].append({
                    'id': tid,
                    'dir': target_dir,
                    'difficulty': difficulty,
                    'rec_u': rec_u,
                    'lig_u': lig_u,
                    'bound': bound if bound.exists() else None,
                    'lig_b': lig_b if lig_b.exists() else None,
                    'partners': partners if partners.exists() else None,
                })

    return targets


# ============================================================
# Single Target Prediction
# ============================================================
def predict_single(model, target_info, device, is_torchscript=True):
    """Predict docking for one target. Returns result dict."""
    tid = target_info['id']

    try:
        # Parse chain info
        rec_chains, lig_chains = (None, None)
        if target_info['partners']:
            rec_chains, lig_chains = parse_partners(target_info['partners'])
        if rec_chains is None:
            rec_chains = get_chains(target_info['rec_u'])
        if lig_chains is None:
            lig_chains = get_chains(target_info['lig_u'])

        # Extract features
        rf = extract_features(str(target_info['rec_u']), rec_chains)
        lf = extract_features(str(target_info['lig_u']), lig_chains)

        # Prepare tensors
        rd = {
            'node_features': torch.tensor(rf['node_features'], dtype=torch.float32).to(device),
            'edge_index': torch.tensor(rf['edges'], dtype=torch.long).to(device),
            'coords': torch.tensor(rf['coords'], dtype=torch.float32).to(device),
        }
        ld = {
            'node_features': torch.tensor(lf['node_features'], dtype=torch.float32).to(device),
            'edge_index': torch.tensor(lf['edges'], dtype=torch.long).to(device),
            'coords': torch.tensor(lf['coords'], dtype=torch.float32).to(device),
        }

        # Forward pass
        with torch.no_grad():
            rot, trans, conf = model(rd, ld)
            pred_lig_coords = apply_rodrigues(ld['coords'], rot, trans)

        pred_np = pred_lig_coords.cpu().numpy()
        confidence = conf.item()

        # Displacement
        displacement = np.linalg.norm(pred_np.mean(0) - lf['coords'].mean(0))

        # TM-scores (if bound structure available)
        tm_complex, tm_ligand = None, None
        if target_info['lig_b'] or target_info['bound']:
            try:
                bound_lig = get_bound_ligand_coords(
                    str(target_info['lig_b'] or target_info['bound']),
                    lig_chains
                )
                min_len = min(len(pred_np), len(bound_lig))
                if min_len > 5:
                    tm_ligand = compute_tm_score(pred_np[:min_len], bound_lig[:min_len])

                    # Full complex TM
                    pred_complex = np.concatenate([rf['coords'], pred_np[:min_len]], axis=0)
                    native_complex = np.concatenate([rf['coords'], bound_lig[:min_len]], axis=0)
                    tm_complex = compute_tm_score(pred_complex, native_complex)
            except Exception:
                pass

        return {
            'success': True,
            'pred_coords': pred_np,
            'rec_chains': rec_chains,
            'lig_chains': lig_chains,
            'confidence': confidence,
            'displacement': displacement,
            'rec_residues': len(rf['coords']),
            'lig_residues': len(lf['coords']),
            'tm_score_complex': tm_complex,
            'tm_score_ligand': tm_ligand,
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Batch prediction + TM-score for DB5.5')
    parser.add_argument('--benchmark', required=True, help='Path to benchmark directory')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', help='Path to TorchScript model (.pt)')
    group.add_argument('--checkpoint', help='Path to training checkpoint (best_model.pt)')
    parser.add_argument('--output', default='predicted_complex', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark)
    output_dir = Path(args.output)
    device = args.device if torch.cuda.is_available() else 'cpu'

    if not benchmark_dir.exists():
        sys.exit(f"Benchmark not found: {benchmark_dir}")

    print("=" * 70)
    print("BATCH PREDICTION + TM-SCORE — DB5.5")
    print("=" * 70)

    # Load model
    is_torchscript = False
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            sys.exit(f"Model not found: {model_path}")
        print(f"  Loading TorchScript model: {model_path}")
        model = torch.jit.load(str(model_path), map_location=device)
        is_torchscript = True
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
        print(f"  Loading checkpoint: {ckpt_path}")
        sys.path.insert(0, str(Path(__file__).parent / '..'))
        sys.path.insert(0, str(Path(__file__).parent))
        from step4_model import ProteinDockingModel
        model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        print(f"  Loaded from epoch {ckpt['epoch']}")

    model.eval()
    print(f"  Device: {device}")

    # Find targets
    targets_by_diff = find_all_targets(benchmark_dir)
    total = sum(len(v) for v in targets_by_diff.values())
    print(f"  Found {total} targets")
    for diff, tlist in targets_by_diff.items():
        print(f"    {diff}: {len(tlist)}")

    # Predict
    results = []
    failed = []

    for difficulty, tlist in targets_by_diff.items():
        print(f"\n  Processing {difficulty} targets...")

        for tinfo in tqdm(tlist, desc=f"    {difficulty}"):
            tid = tinfo['id']
            pred = predict_single(model, tinfo, device, is_torchscript)

            if pred['success']:
                # Save predicted PDB
                out_dir = output_dir / f'{difficulty}_targets' / tid
                out_dir.mkdir(parents=True, exist_ok=True)
                out_pdb = out_dir / f'{tid}_predicted.pdb'

                write_predicted_pdb(
                    str(tinfo['rec_u']), str(tinfo['lig_u']),
                    pred['pred_coords'], pred['rec_chains'], pred['lig_chains'],
                    out_pdb
                )

                results.append({
                    'target': tid,
                    'difficulty': difficulty,
                    'confidence': round(pred['confidence'], 4),
                    'displacement': round(pred['displacement'], 4),
                    'rec_residues': pred['rec_residues'],
                    'lig_residues': pred['lig_residues'],
                    'tm_score_complex': round(pred['tm_score_complex'], 4) if pred['tm_score_complex'] else None,
                    'tm_score_ligand': round(pred['tm_score_ligand'], 4) if pred['tm_score_ligand'] else None,
                    'output_pdb': str(out_pdb),
                    'error': None,
                })
            else:
                failed.append({'target': tid, 'difficulty': difficulty, 'error': pred['error']})
                results.append({
                    'target': tid, 'difficulty': difficulty,
                    'confidence': None, 'displacement': None,
                    'rec_residues': None, 'lig_residues': None,
                    'tm_score_complex': None, 'tm_score_ligand': None,
                    'output_pdb': None, 'error': pred['error'],
                })

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'batch_predictions.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Print summary
    valid = df[df['error'].isna()]

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total: {total}")
    print(f"  Predicted: {len(valid)}")
    print(f"  Failed: {len(failed)}")

    if len(valid) > 0 and valid['tm_score_complex'].notna().sum() > 0:
        tm_valid = valid[valid['tm_score_complex'].notna()]
        print(f"\n  TM-score (complex):")
        print(f"    Mean:   {tm_valid['tm_score_complex'].mean():.4f} +/- {tm_valid['tm_score_complex'].std():.4f}")
        print(f"    Median: {tm_valid['tm_score_complex'].median():.4f}")
        print(f"    TM >= 0.5: {(tm_valid['tm_score_complex'] >= 0.5).sum()}/{len(tm_valid)} "
              f"({(tm_valid['tm_score_complex'] >= 0.5).mean()*100:.1f}%)")
        print(f"    TM >= 0.8: {(tm_valid['tm_score_complex'] >= 0.8).sum()}/{len(tm_valid)} "
              f"({(tm_valid['tm_score_complex'] >= 0.8).mean()*100:.1f}%)")

        print(f"\n  TM-score (ligand):")
        tm_lig = valid[valid['tm_score_ligand'].notna()]
        print(f"    Mean:   {tm_lig['tm_score_ligand'].mean():.4f} +/- {tm_lig['tm_score_ligand'].std():.4f}")
        print(f"    Median: {tm_lig['tm_score_ligand'].median():.4f}")

        print(f"\n  By difficulty:")
        for diff in ['rigid', 'medium', 'difficult']:
            sub = tm_valid[tm_valid['difficulty'] == diff]
            if len(sub) > 0:
                print(f"    {diff:<12}: TM(complex)={sub['tm_score_complex'].mean():.4f}  "
                      f"TM(ligand)={sub['tm_score_ligand'].mean():.4f}  n={len(sub)}")

    if failed:
        print(f"\n  Failed targets:")
        for f in failed[:5]:
            print(f"    {f['target']} ({f['difficulty']}): {f['error'][:60]}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    print(f"\n  CSV saved: {csv_path}")
    print(f"  PDBs saved: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()