#!/usr/bin/env python3
"""
I-RMSD vs DockQ scatter plot colored by difficulty.
Reads from capri_results.csv.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

# Load results
script_dir = Path(__file__).parent
project_root = script_dir.parent
eval_dir = sorted(project_root.glob('experiments/*/evaluation'))[-1]
df = pd.read_csv(eval_dir / 'capri_results.csv')
df = df[df['error'].isna()]

# Map difficulty to numeric for colorbar
diff_map = {'rigid_targets': 0, 'medium_targets': 1, 'difficult_targets': 2}
diff_labels = {'rigid_targets': 'Rigid', 'medium_targets': 'Medium', 'difficult_targets': 'Difficult'}
df['diff_num'] = df['difficulty'].map(diff_map)

# Custom colormap: purple (rigid) → teal (medium) → yellow (difficult)
colors_list = ['#5E3C99', '#2AA198', '#E6C616']
cmap = mcolors.LinearSegmentedColormap.from_list('difficulty', colors_list, N=256)

# LNCS half-width ~6cm ≈ 2.36in, use 2.5 for slight margin
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)

# Scatter plot
sc = ax.scatter(df['irmsd'], df['dockq'],
                c=df['diff_num'], cmap=cmap, vmin=-0.3, vmax=2.3,
                s=12, alpha=0.8, edgecolors='white', linewidths=0.2, zorder=3)

# DockQ threshold line
ax.axhline(y=0.23, color='#cc3333', linestyle='--', linewidth=0.7, alpha=0.8, zorder=2)
ax.text(ax.get_xlim()[1] * 0.65, 0.25, 'DockQ = 0.23', fontsize=4.5, color='#cc3333', va='bottom')

# Colorbar
cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2], pad=0.02, aspect=20, shrink=0.85)
cbar.ax.set_yticklabels(['Rigid', 'Medium', 'Difficult'], fontsize=5, fontweight='bold')
cbar.ax.tick_params(length=0)
cbar.set_label('Difficulty', fontsize=6, fontweight='bold', labelpad=2)
cbar.outline.set_linewidth(0.5)

# Labels
ax.set_xlabel('I-RMSD (Å)', fontsize=7, fontweight='bold', labelpad=2)
ax.set_ylabel('DockQ Score', fontsize=7, fontweight='bold', labelpad=2)
ax.set_title('I-RMSD vs DockQ', fontsize=8, fontweight='bold', pad=4)

# Ticks
ax.tick_params(axis='both', labelsize=5.5)
ax.set_xlim(-0.5, max(df['irmsd'].max() * 1.05, 10))
ax.set_ylim(-0.05, 1.05)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.grid(alpha=0.15, linewidth=0.3)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.3)

out_dir = project_root / 'paper'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'fig_irmsd_dockq.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_dir / 'fig_irmsd_dockq.png', bbox_inches='tight', dpi=300)
print(f'Saved to {out_dir}/fig_irmsd_dockq.pdf')
print(f'Targets: {len(df)}')
print(f'DockQ >= 0.23: {(df["dockq"] >= 0.23).sum()}/{len(df)}')