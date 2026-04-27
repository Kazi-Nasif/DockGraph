#!/usr/bin/env python3
"""
Analyze AlphaRED Figure 5 Source Data (eLife 2025)
Extract exact success rates and quality breakdowns for AFm and AlphaRED.
Source: Harmalkar et al., eLife 13, RP94029 (2025), Figure 5 — source data
https://cdn.elifesciences.org/articles/94029/elife-94029-fig5-data1-v1.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load eLife xlsx — row 0 is sub-header, skip it
script_dir = Path(__file__).parent
project_root = script_dir.parent
xlsx_path = project_root / 'data' / 'alphared_table_s1_elife.xlsx'

raw = pd.read_excel(xlsx_path)

# Row 0 contains sub-headers (AF2, AlphaRED), actual data starts at row 1
df = raw.iloc[1:].copy().reset_index(drop=True)
df.columns = ['PDB', 'IRMSD_AF2', 'IRMSD_AlphaRED', 'fnat_AF2', 'fnat_AlphaRED',
              'CAPRI_AF2', 'CAPRI_AlphaRED', 'Flexibility']

# Convert numeric columns
for col in ['IRMSD_AF2', 'IRMSD_AlphaRED', 'fnat_AF2', 'fnat_AlphaRED',
            'CAPRI_AF2', 'CAPRI_AlphaRED']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows with NaN PDB
df = df[df['PDB'].notna()].reset_index(drop=True)

# Also load BM5.5 Excel to identify Ab-Ag targets
bm_path = project_root / 'data' / 'Table_BM5.5.xlsx'
bm = pd.read_excel(bm_path)
bm['pdb_id'] = bm['Complex'].str.split('_').str[0]
ab_pdb_ids = set(bm[bm['Category'].isin(['AA', 'AS'])]['pdb_id'].values)

# Tag Ab-Ag in the AlphaRED table
df['is_ab'] = df['PDB'].isin(ab_pdb_ids)

print("=" * 70)
print("ALPHARED ELIFE FIGURE 5 SOURCE DATA ANALYSIS")
print("Source: Harmalkar et al., eLife 13, RP94029 (2025)")
print("=" * 70)
print(f"Total targets: {len(df)}")
print(f"Flexibility: {df['Flexibility'].value_counts().to_dict()}")
print(f"Ab-Ag targets identified: {df['is_ab'].sum()}")
print(f"CAPRI rank values (AF2): {sorted(df['CAPRI_AF2'].dropna().unique())}")
print(f"CAPRI rank values (AlphaRED): {sorted(df['CAPRI_AlphaRED'].dropna().unique())}")

# CAPRI Rank: 0=Incorrect, 1=Acceptable, 2=Medium, 3=High

def analyze(df_sub, col, label):
    """Compute quality breakdown."""
    n = len(df_sub)
    if n == 0:
        return
    h = (df_sub[col] == 3).sum()
    m = (df_sub[col] == 2).sum()
    a = (df_sub[col] == 1).sum()
    inc = (df_sub[col] == 0).sum()
    success = h + m + a
    print(f"    {label:<20} (n={n:>3}): "
          f"H={h:>3}({h/n*100:>5.1f}%)  "
          f"M={m:>3}({m/n*100:>5.1f}%)  "
          f"A={a:>3}({a/n*100:>5.1f}%)  "
          f"Inc={inc:>3}({inc/n*100:>5.1f}%)  "
          f"Success={success:>3}/{n}({success/n*100:>5.1f}%)")

for method, col in [('AF-MULTIMER', 'CAPRI_AF2'), ('ALPHARED', 'CAPRI_AlphaRED')]:
    print(f"\n{'='*70}")
    print(f"  {method}")
    print(f"{'='*70}")

    print(f"\n  BY DIFFICULTY:")
    for diff in ['Rigid', 'Medium', 'Difficult']:
        analyze(df[df['Flexibility'] == diff], col, diff)
    analyze(df, col, 'Overall')

    print(f"\n  BY CATEGORY:")
    analyze(df[df['is_ab']], col, 'Ab-Ag')
    analyze(df[~df['is_ab']], col, 'General')

# Print values formatted for bar chart code
print(f"\n{'='*70}")
print("VALUES FOR BAR CHART (copy into plot_comparison.py)")
print(f"{'='*70}")

for method, col in [('AFm', 'CAPRI_AF2'), ('AlphaRED', 'CAPRI_AlphaRED')]:
    print(f"\n# {method}")
    for diff in ['Rigid', 'Medium', 'Difficult']:
        sub = df[df['Flexibility'] == diff]
        n = len(sub)
        h = (sub[col] == 3).sum() / n * 100
        m = (sub[col] == 2).sum() / n * 100
        a = (sub[col] == 1).sum() / n * 100
        total = h + m + a
        print(f"# {diff}: High={h:.1f}%, Medium={m:.1f}%, Acceptable={a:.1f}%, Total={total:.1f}%")

    n = len(df)
    h = (df[col] == 3).sum() / n * 100
    m = (df[col] == 2).sum() / n * 100
    a = (df[col] == 1).sum() / n * 100
    total = h + m + a
    print(f"# Overall: High={h:.1f}%, Medium={m:.1f}%, Acceptable={a:.1f}%, Total={total:.1f}%")

    ab = df[df['is_ab']]
    n_ab = len(ab)
    h = (ab[col] == 3).sum() / n_ab * 100
    m = (ab[col] == 2).sum() / n_ab * 100
    a = (ab[col] == 1).sum() / n_ab * 100
    total = h + m + a
    print(f"# Ab-Ag: High={h:.1f}%, Medium={m:.1f}%, Acceptable={a:.1f}%, Total={total:.1f}%")

    gen = df[~df['is_ab']]
    n_gen = len(gen)
    h = (gen[col] == 3).sum() / n_gen * 100
    m = (gen[col] == 2).sum() / n_gen * 100
    a = (gen[col] == 1).sum() / n_gen * 100
    total = h + m + a
    print(f"# General: High={h:.1f}%, Medium={m:.1f}%, Acceptable={a:.1f}%, Total={total:.1f}%")

# Verification against published text
print(f"\n{'='*70}")
print("VERIFICATION AGAINST ELIFE PAPER TEXT")
print(f"{'='*70}")
ared = (df['CAPRI_AlphaRED'] >= 1).sum()
afm = (df['CAPRI_AF2'] >= 1).sum()
ared_ab = (df[df['is_ab']]['CAPRI_AlphaRED'] >= 1).sum()
afm_ab = (df[df['is_ab']]['CAPRI_AF2'] >= 1).sum()
n_ab = df['is_ab'].sum()
print(f"AlphaRED overall: {ared}/{len(df)} = {ared/len(df)*100:.1f}%  (eLife text: 63%)")
print(f"AFm overall:      {afm}/{len(df)} = {afm/len(df)*100:.1f}%  (eLife text: 43%)")
print(f"AlphaRED Ab-Ag:   {ared_ab}/{n_ab} = {ared_ab/n_ab*100:.1f}%  (eLife text: 43%)")
print(f"AFm Ab-Ag:        {afm_ab}/{n_ab} = {afm_ab/n_ab*100:.1f}%  (eLife text: 20%)")

# Also save as clean CSV for other scripts
out_csv = project_root / 'data' / 'alphared_elife_fig5.csv'
df.to_csv(out_csv, index=False)
print(f"\nCleaned data saved to: {out_csv}")