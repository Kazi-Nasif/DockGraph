#!/usr/bin/env python3
"""Generate DockQ distribution histogram from evaluation results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Load actual results
csv_path = Path(__file__).parent.parent / 'experiments' / '20260307_073713_full_training' / 'evaluation' / 'capri_results.csv'
df = pd.read_csv(csv_path)
df = df[df['error'].isna()]  # drop failed targets

# Map difficulty labels
diff_map = {'rigid_targets': 'Rigid', 'medium_targets': 'Medium', 'difficult_targets': 'Difficult'}
df['diff_label'] = df['difficulty'].map(diff_map)

# LNCS half-width: ~6cm ≈ 2.36 in
fig, ax = plt.subplots(figsize=(2.5, 2.8), dpi=300)

# Colors per difficulty
colors = {'Rigid': '#7570B3', 'Medium': '#1B9E77', 'Difficult': '#D95F02'}

bins = np.arange(0, 1.05, 0.05)

# Stacked histogram
rigid = df[df['diff_label'] == 'Rigid']['dockq'].values
medium = df[df['diff_label'] == 'Medium']['dockq'].values
difficult = df[df['diff_label'] == 'Difficult']['dockq'].values

ax.hist([rigid, medium, difficult], bins=bins, stacked=True,
        color=[colors['Rigid'], colors['Medium'], colors['Difficult']],
        edgecolor='white', linewidth=0.3, zorder=3)

# CAPRI threshold lines
thresholds = [(0.23, 'Acc.'), (0.49, 'Med.'), (0.80, 'High')]
for val, label in thresholds:
    ax.axvline(x=val, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=2)
    ax.text(val + 0.01, ax.get_ylim()[1] * 0.92, label, fontsize=4.5, color='gray', va='top')

# Mean line
mean_dq = df['dockq'].mean()
ax.axvline(x=mean_dq, color='black', linestyle='-', linewidth=0.8, zorder=4)
ax.text(mean_dq - 0.02, ax.get_ylim()[1] * 0.78, f'Mean\n{mean_dq:.3f}', 
        fontsize=4.5, ha='right', fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=colors['Rigid'], label=f'Rigid (n={len(rigid)})'),
    mpatches.Patch(facecolor=colors['Medium'], label=f'Medium (n={len(medium)})'),
    mpatches.Patch(facecolor=colors['Difficult'], label=f'Difficult (n={len(difficult)})'),
]
ax.legend(handles=legend_elements, fontsize=4.5, loc='upper left', framealpha=0.9,
          edgecolor='none', handlelength=1.0, handletextpad=0.3, borderpad=0.3)

ax.set_xlabel('DockQ Score', fontsize=7, labelpad=2)
ax.set_ylabel('Number of Targets', fontsize=7, labelpad=2)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', labelsize=6)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.grid(axis='y', alpha=0.2, linewidth=0.3)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)

out_dir = Path(__file__).parent.parent / 'paper'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'fig_dockq_dist.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_dir / 'fig_dockq_dist.png', bbox_inches='tight', dpi=300)
print(f'Saved to {out_dir}/fig_dockq_dist.pdf')
print(f'Targets: {len(df)} (Rigid={len(rigid)}, Medium={len(medium)}, Difficult={len(difficult)})')
print(f'Mean DockQ: {mean_dq:.4f}')