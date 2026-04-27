#!/usr/bin/env python3
"""
Three comparison visualization options:
1. Scatter plots (DockGraph vs baselines per-target)
2. Box/violin plots (I-RMSD distribution)
3. Improvement table (biggest gains)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent
eval_dir = sorted(project_root.glob('experiments/*/evaluation'))[-1]

# Load DockGraph results
dg = pd.read_csv(eval_dir / 'capri_results.csv')
dg = dg[dg['error'].isna()]
dg['pdb_id'] = dg['target'].str.split('_').str[0]

# Load AlphaRED Table S1
ared = pd.read_csv(project_root / 'data' / 'alphared_elife_fig5.csv')

# Merge on PDB ID
merged = pd.merge(dg, ared, left_on='pdb_id', right_on='PDB', how='inner')
print(f"Merged targets: {len(merged)}")
print(f"Columns: {merged.columns.tolist()}")

# Colors
c_dg = '#1a5276'
c_ar = '#c0392b'
c_af = '#f39c12'

# ================================================================
# OPTION 1: Scatter plots (like AlphaRED Figure 5)
# ================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(4.8, 2.5), dpi=300)

# Color by difficulty
diff_colors = {'Rigid': '#5E3C99', 'Medium': '#2AA198', 'Difficult': '#E6C616'}
merged['color'] = merged['Flexibility'].map(diff_colors)

# 1a: I-RMSD — DockGraph (x) vs AlphaRED (y)
ax = axes1[0]
for diff, color in diff_colors.items():
    sub = merged[merged['Flexibility'] == diff]
    ax.scatter(sub['irmsd'], sub['IRMSD_AlphaRED'], c=color, s=10, alpha=0.7,
               edgecolors='white', linewidths=0.2, label=diff, zorder=3)

# Diagonal line
lim = max(merged['irmsd'].max(), merged['IRMSD_AlphaRED'].max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.5, alpha=0.5, zorder=2)
ax.set_xlabel('I-RMSD DockGraph (Å)', fontsize=7, fontweight='bold')
ax.set_ylabel('I-RMSD AlphaRED (Å)', fontsize=7, fontweight='bold')
ax.set_title('I-RMSD Comparison', fontsize=7, fontweight='bold')
ax.tick_params(labelsize=5.5)
ax.set_xlim(0, min(lim, 25))
ax.set_ylim(0, min(lim, 25))

# Add text: points below diagonal = AlphaRED better
below = (merged['irmsd'] > merged['IRMSD_AlphaRED']).sum()
above = (merged['irmsd'] < merged['IRMSD_AlphaRED']).sum()
ax.text(0.95, 0.05, f'DockGraph\nbetter\n({above})', transform=ax.transAxes,
        fontsize=5, ha='right', va='bottom', color=c_dg, fontweight='bold')
ax.text(0.05, 0.95, f'AlphaRED\nbetter\n({below})', transform=ax.transAxes,
        fontsize=5, ha='left', va='top', color=c_ar, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# 1b: fnat — DockGraph (x) vs AlphaRED (y)
ax = axes1[1]
for diff, color in diff_colors.items():
    sub = merged[merged['Flexibility'] == diff]
    ax.scatter(sub['fnat'], sub['fnat_AlphaRED'], c=color, s=10, alpha=0.7,
               edgecolors='white', linewidths=0.2, label=diff, zorder=3)

ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.5, zorder=2)
ax.set_xlabel('fnat DockGraph', fontsize=7, fontweight='bold')
ax.set_ylabel('fnat AlphaRED', fontsize=7, fontweight='bold')
ax.set_title('fnat Comparison', fontsize=7, fontweight='bold')
ax.tick_params(labelsize=5.5)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

below_f = (merged['fnat'] < merged['fnat_AlphaRED']).sum()
above_f = (merged['fnat'] > merged['fnat_AlphaRED']).sum()
ax.text(0.95, 0.05, f'DockGraph\nbetter\n({above_f})', transform=ax.transAxes,
        fontsize=5, ha='right', va='bottom', color=c_dg, fontweight='bold')
ax.text(0.05, 0.95, f'AlphaRED\nbetter\n({below_f})', transform=ax.transAxes,
        fontsize=5, ha='left', va='top', color=c_ar, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# Legend
legend_elements = [mpatches.Patch(facecolor=c, label=d) for d, c in diff_colors.items()]
fig1.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=6,
            frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(pad=0.4)
plt.subplots_adjust(top=0.88)

out_dir = project_root / 'paper'
out_dir.mkdir(exist_ok=True)
fig1.savefig(out_dir / 'option1_scatter.pdf', bbox_inches='tight', dpi=300)
fig1.savefig(out_dir / 'option1_scatter.png', bbox_inches='tight', dpi=300)
print("Option 1 saved: option1_scatter.pdf")

# ================================================================
# OPTION 2: Box plots (I-RMSD and fnat distribution)
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(4.8, 2.5), dpi=300)

# 2a: I-RMSD box plot
ax = axes2[0]
data_irmsd = [
    merged['irmsd'].values,
    merged['IRMSD_AlphaRED'].values,
    merged['IRMSD_AF2'].values,
]
bp = ax.boxplot(data_irmsd, labels=['DockGraph', 'AlphaRED', 'AFm'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', lw=1),
                flierprops=dict(marker='.', markersize=2, alpha=0.5),
                whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7))
colors_box = [c_dg, c_ar, c_af]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('I-RMSD (Å)', fontsize=7, fontweight='bold')
ax.set_title('I-RMSD Distribution', fontsize=7, fontweight='bold')
ax.tick_params(labelsize=6)
ax.set_ylim(0, min(merged[['irmsd', 'IRMSD_AlphaRED', 'IRMSD_AF2']].max().max() * 1.1, 30))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.grid(axis='y', alpha=0.15, lw=0.3)
ax.set_axisbelow(True)

# Add median labels
for i, d in enumerate(data_irmsd):
    med = np.median(d)
    ax.text(i+1, med + 0.3, f'{med:.2f}', ha='center', fontsize=5, fontweight='bold',
            color=colors_box[i])

# 2b: fnat box plot
ax = axes2[1]
data_fnat = [
    merged['fnat'].values,
    merged['fnat_AlphaRED'].values,
    merged['fnat_AF2'].values,
]
bp = ax.boxplot(data_fnat, labels=['DockGraph', 'AlphaRED', 'AFm'],
                patch_artist=True, widths=0.5,
                medianprops=dict(color='black', lw=1),
                flierprops=dict(marker='.', markersize=2, alpha=0.5),
                whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('fnat', fontsize=7, fontweight='bold')
ax.set_title('fnat Distribution', fontsize=7, fontweight='bold')
ax.tick_params(labelsize=6)
ax.set_ylim(-0.05, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.grid(axis='y', alpha=0.15, lw=0.3)
ax.set_axisbelow(True)

for i, d in enumerate(data_fnat):
    med = np.median(d)
    ax.text(i+1, med + 0.03, f'{med:.2f}', ha='center', fontsize=5, fontweight='bold',
            color=colors_box[i])

plt.tight_layout(pad=0.4)
fig2.savefig(out_dir / 'option2_boxplot.pdf', bbox_inches='tight', dpi=300)
fig2.savefig(out_dir / 'option2_boxplot.png', bbox_inches='tight', dpi=300)
print("Option 2 saved: option2_boxplot.pdf")

# ================================================================
# OPTION 3: Improvement table — targets where baselines fail, DockGraph succeeds
# ================================================================
# Define success: CAPRI >= 1 for baselines, DockQ >= 0.23 for DockGraph
merged['dg_success'] = merged['dockq'] >= 0.23
merged['ar_fail'] = merged['CAPRI_AlphaRED'] == 0
merged['af_fail'] = merged['CAPRI_AF2'] == 0

# Targets where AlphaRED failed but DockGraph succeeded
rescued_ar = merged[merged['ar_fail'] & merged['dg_success']].sort_values('dockq', ascending=False)

# Targets where AFm failed but DockGraph succeeded
rescued_af = merged[merged['af_fail'] & merged['dg_success']].sort_values('dockq', ascending=False)

print(f"\n{'='*70}")
print("OPTION 3: IMPROVEMENT TABLE")
print(f"{'='*70}")
print(f"\nTargets where AlphaRED FAILED but DockGraph SUCCEEDED: {len(rescued_ar)}")
print(f"  {'PDB':<6} {'Flex':<12} {'DG DockQ':>10} {'DG I-RMSD':>10} {'AR I-RMSD':>10} {'AF I-RMSD':>10}")
print(f"  {'-'*62}")
for _, r in rescued_ar.head(15).iterrows():
    print(f"  {r['pdb_id']:<6} {r['Flexibility']:<12} {r['dockq']:>10.4f} {r['irmsd']:>10.2f} "
          f"{r['IRMSD_AlphaRED']:>10.2f} {r['IRMSD_AF2']:>10.2f}")

print(f"\nTargets where AFm FAILED but DockGraph SUCCEEDED: {len(rescued_af)}")
print(f"  {'PDB':<6} {'Flex':<12} {'DG DockQ':>10} {'DG I-RMSD':>10} {'AF I-RMSD':>10} {'AR I-RMSD':>10}")
print(f"  {'-'*62}")
for _, r in rescued_af.head(15).iterrows():
    print(f"  {r['pdb_id']:<6} {r['Flexibility']:<12} {r['dockq']:>10.4f} {r['irmsd']:>10.2f} "
          f"{r['IRMSD_AF2']:>10.2f} {r['IRMSD_AlphaRED']:>10.2f}")

# Both failed but DockGraph succeeded
both_fail = merged[merged['ar_fail'] & merged['af_fail'] & merged['dg_success']]
print(f"\nBoth AlphaRED AND AFm FAILED but DockGraph SUCCEEDED: {len(both_fail)}")
print(f"  {'PDB':<6} {'Flex':<12} {'DG DockQ':>10} {'DG I-RMSD':>10} {'AR I-RMSD':>10} {'AF I-RMSD':>10}")
print(f"  {'-'*62}")
for _, r in both_fail.sort_values('dockq', ascending=False).iterrows():
    print(f"  {r['pdb_id']:<6} {r['Flexibility']:<12} {r['dockq']:>10.4f} {r['irmsd']:>10.2f} "
          f"{r['IRMSD_AlphaRED']:>10.2f} {r['IRMSD_AF2']:>10.2f}")

# Summary stats
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")
print(f"  Total matched targets: {len(merged)}")
print(f"  DockGraph success: {merged['dg_success'].sum()}/{len(merged)} ({merged['dg_success'].mean()*100:.1f}%)")
print(f"  AlphaRED success: {(merged['CAPRI_AlphaRED']>=1).sum()}/{len(merged)} ({(merged['CAPRI_AlphaRED']>=1).mean()*100:.1f}%)")
print(f"  AFm success: {(merged['CAPRI_AF2']>=1).sum()}/{len(merged)} ({(merged['CAPRI_AF2']>=1).mean()*100:.1f}%)")
print(f"\n  Median I-RMSD: DG={merged['irmsd'].median():.2f}  AR={merged['IRMSD_AlphaRED'].median():.2f}  AFm={merged['IRMSD_AF2'].median():.2f}")
print(f"  Median fnat:   DG={merged['fnat'].median():.2f}  AR={merged['fnat_AlphaRED'].median():.2f}  AFm={merged['fnat_AF2'].median():.2f}")

# DockGraph has lower I-RMSD than AlphaRED in how many targets?
dg_better_irmsd_ar = (merged['irmsd'] < merged['IRMSD_AlphaRED']).sum()
dg_better_irmsd_af = (merged['irmsd'] < merged['IRMSD_AF2']).sum()
dg_better_fnat_ar = (merged['fnat'] > merged['fnat_AlphaRED']).sum()
dg_better_fnat_af = (merged['fnat'] > merged['fnat_AF2']).sum()
print(f"\n  DockGraph has lower I-RMSD than AlphaRED: {dg_better_irmsd_ar}/{len(merged)} ({dg_better_irmsd_ar/len(merged)*100:.1f}%)")
print(f"  DockGraph has lower I-RMSD than AFm:      {dg_better_irmsd_af}/{len(merged)} ({dg_better_irmsd_af/len(merged)*100:.1f}%)")
print(f"  DockGraph has higher fnat than AlphaRED:   {dg_better_fnat_ar}/{len(merged)} ({dg_better_fnat_ar/len(merged)*100:.1f}%)")
print(f"  DockGraph has higher fnat than AFm:        {dg_better_fnat_af}/{len(merged)} ({dg_better_fnat_af/len(merged)*100:.1f}%)")

print(f"\nAll figures saved to: {out_dir}/")