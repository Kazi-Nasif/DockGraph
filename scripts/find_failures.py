#!/usr/bin/env python3
"""
Find all DockGraph failure cases (DockQ < 0.23)
and cross-reference with AlphaRED eLife source data.
"""

import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
eval_dir = sorted(project_root.glob('experiments/*/evaluation'))[-1]

# Load DockGraph results
dg = pd.read_csv(eval_dir / 'capri_results.csv')
dg['pdb_id'] = dg['target'].str.split('_').str[0]

# Load eLife baseline data
elife = pd.read_csv(project_root / 'data' / 'alphared_elife_fig5.csv')

# All failed targets (DockQ < 0.23 OR error)
failed = dg[(dg['dockq'] < 0.23) | (dg['error'].notna())].copy()

print("=" * 70)
print("ALL DOCKGRAPH FAILURE CASES")
print("=" * 70)
print(f"Total evaluated: {len(dg)}")
print(f"Total with errors: {dg['error'].notna().sum()}")
print(f"Total DockQ < 0.23 (excluding errors): {((dg['dockq'] < 0.23) & dg['error'].isna()).sum()}")
print(f"Total failures: {len(failed)}")

print(f"\n{'PDB':<8} {'Target':<20} {'Diff':<18} {'DockQ':>8} {'I-RMSD':>8} {'fnat':>8} {'Error'}")
print("-" * 90)

for _, row in failed.sort_values('dockq', ascending=True).iterrows():
    pdb = row['pdb_id']
    target = row['target'] if pd.notna(row['target']) else '—'
    diff = row['difficulty'] if pd.notna(row['difficulty']) else '—'
    
    if pd.notna(row.get('error')):
        print(f"{pdb:<8} {target:<20} {diff:<18} {'—':>8} {'—':>8} {'—':>8} {row['error'][:40]}")
    else:
        print(f"{pdb:<8} {target:<20} {diff:<18} {row['dockq']:>8.4f} {row['irmsd']:>8.2f} {row['fnat']:>8.4f}")

# Cross-reference with eLife data
print(f"\n{'=' * 70}")
print("CROSS-REFERENCE WITH ELIFE BASELINE DATA")
print("=" * 70)
print(f"\n{'PDB':<8} {'Diff':<15} {'DG DockQ':>10} {'DG IRMSD':>10} {'AR IRMSD':>10} {'AFm IRMSD':>10} {'AR CAPRI':>10} {'AFm CAPRI':>10}")
print("-" * 85)

for _, row in failed.sort_values('dockq', ascending=True).iterrows():
    pdb = row['pdb_id']
    
    # Find in eLife data
    match = elife[elife['PDB'] == pdb]
    
    if len(match) > 0:
        m = match.iloc[0]
        dockq = f"{row['dockq']:.4f}" if pd.notna(row.get('dockq')) else '—'
        irmsd = f"{row['irmsd']:.2f}" if pd.notna(row.get('irmsd')) else '—'
        diff = row['difficulty'] if pd.notna(row.get('difficulty')) else '—'
        print(f"{pdb:<8} {diff:<15} {dockq:>10} {irmsd:>10} {m['IRMSD_AlphaRED']:>10.2f} {m['IRMSD_AF2']:>10.2f} {int(m['CAPRI_AlphaRED']):>10} {int(m['CAPRI_AF2']):>10}")
    else:
        print(f"{pdb:<8} {'—':<15} {'—':>10} {'—':>10} {'not in eLife data':>30}")

# Also check: how many of our failures also failed for baselines?
print(f"\n{'=' * 70}")
print("BASELINE PERFORMANCE ON OUR FAILURE CASES")
print("=" * 70)

for _, row in failed.sort_values('dockq', ascending=True).iterrows():
    pdb = row['pdb_id']
    match = elife[elife['PDB'] == pdb]
    if len(match) > 0:
        m = match.iloc[0]
        ar_status = "FAILED" if m['CAPRI_AlphaRED'] == 0 else f"CAPRI={int(m['CAPRI_AlphaRED'])}"
        af_status = "FAILED" if m['CAPRI_AF2'] == 0 else f"CAPRI={int(m['CAPRI_AF2'])}"
        print(f"  {pdb}: AlphaRED {ar_status}, AFm {af_status}")