#!/usr/bin/env python3
"""
Evaluate DockGraph on Antibody-Antigen (AA + AS) targets from DB5.5.
Reads target list directly from Table_BM5.5.xlsx.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
from tqdm import tqdm

from step2_data_loader import ProteinDockingLoader
from step3_feature_extraction import ProteinFeatureExtractor
from step4_model import ProteinDockingModel, prepare_graph_data
from step6_evaluate import evaluate_single

DIFF_MAP = {
    'Rigid-body': 'rigid_targets',
    'Medium': 'medium_targets',
    'Difficult': 'difficult_targets',
}


def load_ab_targets(xlsx_path):
    """Load AA and AS targets from the benchmark Excel file."""
    df = pd.read_excel(xlsx_path)
    ab = df[df['Category'].isin(['AA', 'AS'])].copy()
    ab['difficulty_dir'] = ab['Difficulty'].map(DIFF_MAP)
    ab['pdb_id'] = ab['Complex'].str.split('_').str[0]
    print(f"  Loaded {len(ab)} antibody-antigen targets from Excel")
    print(f"    AA: {len(ab[ab['Category']=='AA'])}, AS: {len(ab[ab['Category']=='AS'])}")
    print(f"    Rigid: {len(ab[ab['Difficulty']=='Rigid-body'])}, "
          f"Medium: {len(ab[ab['Difficulty']=='Medium'])}, "
          f"Difficult: {len(ab[ab['Difficulty']=='Difficult'])}")
    return ab


def find_target_dir(pdb_id, difficulty_dir, benchmark_path):
    """Find target directory by PDB ID prefix."""
    diff_path = benchmark_path / difficulty_dir
    if not diff_path.exists():
        return None
    for d in diff_path.iterdir():
        if d.is_dir() and (d.name == pdb_id or d.name.startswith(pdb_id)):
            return d.name
    return None


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    xlsx_path = project_root / 'data' / 'Table_BM5.5.xlsx'

    if not xlsx_path.exists():
        print(f"Excel file not found: {xlsx_path}")
        return

    candidates = sorted(project_root.glob('experiments/*/best_model.pt'))
    if not candidates:
        print("No checkpoint found.")
        return
    ckpt = candidates[-1]

    print("=" * 70)
    print("ANTIBODY-ANTIGEN EVALUATION")
    print("=" * 70)
    print(f"  Excel: {xlsx_path.name}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device: {device}")

    ab_df = load_ab_targets(xlsx_path)

    model = ProteinDockingModel(node_features=26, hidden_dim=128, num_layers=3)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    print(f"  Model loaded from epoch {checkpoint['epoch']}")

    loader = ProteinDockingLoader()
    extractor = ProteinFeatureExtractor()
    benchmark_path = Path(loader.benchmark_path)

    rows = []
    not_found = []

    for _, entry in tqdm(ab_df.iterrows(), total=len(ab_df), desc="  Evaluating"):
        pdb_id = entry['pdb_id']
        complex_name = entry['Complex']
        category = entry['Category']
        difficulty = entry['Difficulty']
        diff_dir = entry['difficulty_dir']

        target_name = find_target_dir(pdb_id, diff_dir, benchmark_path)

        if target_name is None:
            not_found.append(pdb_id)
            rows.append({
                'pdb_id': pdb_id, 'complex': complex_name, 'category': category,
                'difficulty': difficulty,
                'dockq': None, 'irmsd': None, 'lrmsd': None, 'fnat': None,
                'quality': None, 'success': False, 'confidence': None,
                'error': 'not found',
            })
            continue

        try:
            features = extractor.extract_full_features(target_name, diff_dir)
            result = evaluate_single(model, features, device)
        except Exception as e:
            result = dict(irmsd=None, lrmsd=None, fnat=None, dockq=None,
                          quality=None, success=False, confidence=None,
                          error=str(e))

        rows.append({
            'pdb_id': pdb_id, 'complex': complex_name, 'category': category,
            'difficulty': difficulty, **result,
        })

    df = pd.DataFrame(rows)
    valid = df[df['error'].isna()]
    total = len(valid)

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Targets in Excel: {len(ab_df)}")
    print(f"  Evaluated: {total}")
    if not_found:
        print(f"  Not found ({len(not_found)}): {not_found}")

    if total == 0:
        print("  No valid results.")
        return

    successes = int(valid['success'].sum())

    print(f"\n  OVERALL (AA + AS, n={total}):")
    print(f"    Success (DockQ >= 0.23): {successes}/{total} ({successes/total*100:.1f}%)")
    for label, thresh in [('High', 0.8), ('Medium', 0.49), ('Acceptable', 0.23)]:
        n = int((valid['dockq'] >= thresh).sum())
        print(f"    {label:>12} (DockQ >= {thresh}): {n} ({n/total*100:.1f}%)")
    print(f"    Mean DockQ:  {valid['dockq'].mean():.4f} +/- {valid['dockq'].std():.4f}")
    print(f"    Mean I-RMSD: {valid['irmsd'].mean():.4f} +/- {valid['irmsd'].std():.4f} A")
    print(f"    Mean L-RMSD: {valid['lrmsd'].mean():.4f} +/- {valid['lrmsd'].std():.4f} A")
    print(f"    Mean fnat:   {valid['fnat'].mean():.4f} +/- {valid['fnat'].std():.4f}")

    for cat, label in [('AA', 'Antibody-Antigen'), ('AS', 'Nanobody-Antigen')]:
        sub = valid[valid['category'] == cat]
        if len(sub) == 0:
            continue
        s = int(sub['success'].sum())
        print(f"\n  {cat} ({label}, n={len(sub)}):")
        print(f"    Success: {s}/{len(sub)} ({s/len(sub)*100:.1f}%)")
        print(f"    Mean DockQ:  {sub['dockq'].mean():.4f}")
        print(f"    Mean I-RMSD: {sub['irmsd'].mean():.4f} A")
        print(f"    Mean fnat:   {sub['fnat'].mean():.4f}")

    print(f"\n  BY DIFFICULTY:")
    for diff in ['Rigid-body', 'Medium', 'Difficult']:
        sub = valid[valid['difficulty'] == diff]
        if len(sub) == 0:
            continue
        s = int(sub['success'].sum())
        print(f"    {diff:<12}: {s}/{len(sub)} ({s/len(sub)*100:.1f}%) success, "
              f"DockQ={sub['dockq'].mean():.4f}, I-RMSD={sub['irmsd'].mean():.4f} A")

    print(f"\n{'=' * 70}")
    print("COMPARISON WITH PUBLISHED BASELINES (Ab-Ag targets)")
    print(f"{'=' * 70}")
    our_rate = successes / total * 100
    print(f"  DockGraph (ours):     {our_rate:.1f}% ({successes}/{total})")
    print(f"  AlphaRED:             43.0% (67 targets)")
    print(f"  AlphaFold-Multimer:   20.0% (67 targets)")
    print(f"  Difference vs AlphaRED: {our_rate - 43.0:+.1f} pp")

    print(f"\n{'=' * 70}")
    print("PER-TARGET DETAILS")
    print(f"{'=' * 70}")
    print(f"  {'PDB':<6} {'Cat':<4} {'Diff':<12} {'DockQ':>7} {'I-RMSD':>8} {'fnat':>7} {'Quality':<12}")
    print(f"  {'-'*62}")
    for _, row in valid.sort_values('dockq', ascending=False).iterrows():
        print(f"  {row['pdb_id']:<6} {row['category']:<4} {row['difficulty']:<12} "
              f"{row['dockq']:>7.4f} {row['irmsd']:>8.4f} {row['fnat']:>7.4f} {row['quality']:<12}")

    failed = valid[~valid['success']]
    if len(failed) > 0:
        print(f"\n  FAILED (DockQ < 0.23):")
        for _, row in failed.iterrows():
            print(f"    {row['pdb_id']}: DockQ={row['dockq']:.4f}, I-RMSD={row['irmsd']:.4f}")

    errors = df[df['error'].notna()]
    if len(errors) > 0:
        print(f"\n  ERRORS ({len(errors)}):")
        for _, row in errors.iterrows():
            print(f"    {row['pdb_id']}: {row['error'][:60]}")

    out_dir = ckpt.parent / 'evaluation'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'antibody_antigen_results.csv', index=False)

    summary = {
        'total_ab_targets': len(ab_df),
        'evaluated': total,
        'success_count': successes,
        'success_rate': round(our_rate, 2),
        'mean_dockq': round(float(valid['dockq'].mean()), 4),
        'mean_irmsd': round(float(valid['irmsd'].mean()), 4),
        'mean_lrmsd': round(float(valid['lrmsd'].mean()), 4),
        'mean_fnat': round(float(valid['fnat'].mean()), 4),
    }
    with open(out_dir / 'antibody_antigen_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()