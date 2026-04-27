#!/usr/bin/env python3
"""
Comparison figure: DockGraph vs AlphaRED vs AFm
All baseline numbers from AlphaRED Supplementary Table S1 (Harmalkar et al., eLife 2025)
Panel A: By difficulty | Panel B: General vs Ab-Ag vs Overall
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

capri = pd.read_csv(eval_dir / 'capri_results.csv')
capri = capri[capri['error'].isna()]
ab_ag = pd.read_csv(eval_dir / 'antibody_antigen_results.csv')
ab_ag = ab_ag[ab_ag['error'].isna()]
ab_pdb_ids = set(ab_ag['pdb_id'].values)
capri['pdb_id'] = capri['target'].str.split('_').str[0]
capri['is_ab'] = capri['pdb_id'].isin(ab_pdb_ids)

def qb(df):
    n = len(df)
    if n == 0: return 0, 0, 0
    h = (df['dockq'] >= 0.8).sum() / n * 100
    m = ((df['dockq'] >= 0.49) & (df['dockq'] < 0.8)).sum() / n * 100
    a = ((df['dockq'] >= 0.23) & (df['dockq'] < 0.49)).sum() / n * 100
    return h, m, a

# DockGraph (from our evaluation)
dg_rigid = qb(capri[capri['difficulty'] == 'rigid_targets'])
dg_medium = qb(capri[capri['difficulty'] == 'medium_targets'])
dg_difficult = qb(capri[capri['difficulty'] == 'difficult_targets'])
dg_general = qb(capri[~capri['is_ab']])
dg_abag = qb(capri[capri['is_ab']])
dg_overall = qb(capri)

# AlphaRED (Supplementary Table S1, Harmalkar et al. eLife 2025)
ar_rigid     = (7.5, 37.7, 23.3)
ar_medium    = (1.7, 32.2, 27.1)
ar_difficult = (0.0, 34.3, 31.4)
ar_general   = (7.0, 44.1, 21.5)
ar_abag      = (0.0, 13.4, 35.8)
ar_overall   = (5.1, 36.0, 25.3)

# AF-multimer (Supplementary Table S1)
af_rigid     = (5.0, 34.6, 10.7)
af_medium    = (1.7, 28.8, 15.3)
af_difficult = (0.0, 37.1, 22.9)
af_general   = (4.8, 41.9, 15.1)
af_abag      = (0.0, 10.4, 9.0)
af_overall   = (3.6, 33.6, 13.4)

print("Quality breakdown (High, Medium, Acceptable):")
for name, dg, ar, af in [
    ('Rigid', dg_rigid, ar_rigid, af_rigid),
    ('Medium', dg_medium, ar_medium, af_medium),
    ('Difficult', dg_difficult, ar_difficult, af_difficult),
    ('General', dg_general, ar_general, af_general),
    ('Ab-Ag', dg_abag, ar_abag, af_abag),
    ('Overall', dg_overall, ar_overall, af_overall),
]:
    print(f"  {name:<12} DG={sum(dg):.1f}%  AR={sum(ar):.1f}%  AFm={sum(af):.1f}%")

# Colors
c_dg_high = '#1a5276';  c_dg_med = '#2980b9';  c_dg_acc = '#85c1e9'
c_ar_high = '#922b21';  c_ar_med = '#c0392b';  c_ar_acc = '#f1948a'
c_af_high = '#b7950b';  c_af_med = '#f39c12';  c_af_acc = '#f9e79f'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 2.2), dpi=300,
                                gridspec_kw={'width_ratios': [3, 3]})
bw = 0.28

def draw_stacked(ax, x, data, ch, cm, ca, bw):
    h, m, a = data
    ax.bar(x, a, bw, color=ca, edgecolor='white', lw=0.3, zorder=3)
    ax.bar(x, m, bw, bottom=a, color=cm, edgecolor='white', lw=0.3, zorder=3)
    ax.bar(x, h, bw, bottom=a+m, color=ch, edgecolor='white', lw=0.3, zorder=3)
    return h + m + a

# Panel A: By difficulty
cats_a = ['Rigid\n(159)', 'Medium\n(59)', 'Difficult\n(35)']
x_a = np.arange(len(cats_a))
dg_a = [dg_rigid, dg_medium, dg_difficult]
ar_a = [ar_rigid, ar_medium, ar_difficult]
af_a = [af_rigid, af_medium, af_difficult]

for i in range(3):
    t = draw_stacked(ax1, x_a[i]-bw, dg_a[i], c_dg_high, c_dg_med, c_dg_acc, bw)
    draw_stacked(ax1, x_a[i], ar_a[i], c_ar_high, c_ar_med, c_ar_acc, bw)
    draw_stacked(ax1, x_a[i]+bw, af_a[i], c_af_high, c_af_med, c_af_acc, bw)
    ax1.text(x_a[i]-bw, t+1.5, f'{t:.0f}', ha='center', va='bottom',
             fontsize=6, fontweight='bold', color=c_dg_high)

ax1.set_xticks(x_a)
ax1.set_xticklabels(cats_a, fontsize=7, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=8, fontweight='bold', labelpad=2)
ax1.set_ylim(0, 110)
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.tick_params(axis='y', labelsize=7)
ax1.tick_params(axis='x', length=0)
ax1.set_xlim(-0.5, len(cats_a)-0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.grid(axis='y', alpha=0.15, lw=0.3)
ax1.set_axisbelow(True)

# Panel B: General vs Ab-Ag vs Overall
cats_b = ['General\n(186)', 'Ab-Ag\n(67)', 'DB5.5\n(253)']
x_b = np.arange(len(cats_b))
dg_b = [dg_general, dg_abag, dg_overall]
ar_b = [ar_general, ar_abag, ar_overall]
af_b = [af_general, af_abag, af_overall]

for i in range(3):
    t = draw_stacked(ax2, x_b[i]-bw, dg_b[i], c_dg_high, c_dg_med, c_dg_acc, bw)
    draw_stacked(ax2, x_b[i], ar_b[i], c_ar_high, c_ar_med, c_ar_acc, bw)
    draw_stacked(ax2, x_b[i]+bw, af_b[i], c_af_high, c_af_med, c_af_acc, bw)
    ax2.text(x_b[i]-bw, t+1.5, f'{t:.0f}', ha='center', va='bottom',
             fontsize=6, fontweight='bold', color=c_dg_high)

ax2.set_xticks(x_b)
ax2.set_xticklabels(cats_b, fontsize=7, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.tick_params(axis='y', labelsize=7)
ax2.tick_params(axis='x', length=0)
ax2.set_xlim(-0.5, len(cats_b)-0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.grid(axis='y', alpha=0.15, lw=0.3)
ax2.set_axisbelow(True)

# ── Legend: single-row, each method shown as 3 adjacent color patches ──────
# Layout: [■■■ DockGraph]  [■■■ AlphaRED]  [■■■ AF-multimer]  High|Medium|Acceptable
# where ■■■ = dark|mid|light for that method's palette

from matplotlib.offsetbox import (AnchoredOffsetbox, HPacker, VPacker,
                                   DrawingArea, TextArea)

def make_method_entry(label, c_high, c_mid, c_acc, pw=8, ph=8, gap=1):
    """Three tiny patches (H|M|A) + label text, packed horizontally."""
    boxes = []
    for color in [c_high, c_mid, c_acc]:
        da = DrawingArea(pw, ph, 0, 0)
        da.add_artist(mpatches.FancyBboxPatch(
            (0, 0), pw, ph, boxstyle='square,pad=0',
            facecolor=color, edgecolor='white', linewidth=0.3))
        boxes.append(da)
    txt = TextArea(f' {label}', textprops=dict(fontsize=6.5, fontweight='bold'))
    boxes.append(txt)
    return HPacker(children=boxes, align='center', pad=0, sep=gap)

def make_tier_label(label, pw=8, ph=8):
    """Single grey swatch + tier name for the key."""
    da = DrawingArea(pw, ph, 0, 0)
    # No swatch, just text arrow
    txt = TextArea(label, textprops=dict(fontsize=6, color='#555'))
    return txt

# Build the row: methods + separator + tier key
method_dg  = make_method_entry('DockGraph',    c_dg_high, c_dg_med, c_dg_acc)
method_ar  = make_method_entry('AlphaRED',     c_ar_high, c_ar_med, c_ar_acc)
method_af  = make_method_entry('AF-multimer',  c_af_high, c_af_med, c_af_acc)

# Top row: method color patches
top_row = HPacker(children=[method_dg, method_ar, method_af],
                  align='center', pad=0, sep=12)

# Bottom row: tier key centered beneath
tier_key = TextArea('(dark\u2192light: High | Medium | Acceptable)',
                    textprops=dict(fontsize=5.5, color='#555', style='italic'))

legend_block = VPacker(children=[top_row, tier_key],
                       align='center', pad=0, sep=2)

anchored = AnchoredOffsetbox(
    loc='upper center', child=legend_block, frameon=False,
    bbox_to_anchor=(0.5, 1.02), bbox_transform=fig.transFigure,
    pad=0, borderpad=0)
fig.add_artist(anchored)

plt.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.88, wspace=0.22)

out_dir = project_root / 'paper'
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'fig_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_dir / 'fig_comparison.png', bbox_inches='tight', dpi=300)
print(f"\nSaved to {out_dir}/fig_comparison.pdf")