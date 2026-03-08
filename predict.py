#!/usr/bin/env python3
"""
DockGraph — Protein-Protein Docking Prediction
===============================================
Usage:
    python predict.py receptor.pdb ligand.pdb
    python predict.py receptor.pdb ligand.pdb --bound bound.pdb --lig_chains C
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Structure, Model
from scipy.spatial.distance import cdist

# ============================================================
# Amino acid encoding (standard biochemistry, not proprietary)
# ============================================================
_AA_PROPS = {
    'ALA': [0,0,0], 'ARG': [0,1,0], 'ASN': [0,0,0], 'ASP': [0,0,1],
    'CYS': [0,0,0], 'GLN': [0,0,0], 'GLU': [0,0,1], 'GLY': [0,0,0],
    'HIS': [0,1,0], 'ILE': [1,0,0], 'LEU': [1,0,0], 'LYS': [0,1,0],
    'MET': [1,0,0], 'PHE': [1,0,0], 'PRO': [0,0,0], 'SER': [0,0,0],
    'THR': [0,0,0], 'TRP': [1,0,0], 'TYR': [1,0,0], 'VAL': [1,0,0],
}
_AA_IDX = {aa: i for i, aa in enumerate(sorted(_AA_PROPS))}


def _extract(pdb_path, chains=None):
    s = PDBParser(QUIET=True).get_structure('p', pdb_path)
    coords, oh, props = [], [], []
    for ch in next(s.get_models()):
        if chains and ch.id not in chains: continue
        for r in ch:
            if r.id[0] != ' ' or 'CA' not in r: continue
            coords.append(r['CA'].get_coord())
            o = [0.0]*20
            if r.resname in _AA_IDX: o[_AA_IDX[r.resname]] = 1.0
            oh.append(o)
            props.append(_AA_PROPS.get(r.resname, [0,0,0]))
    c = np.array(coords, dtype=np.float32)
    nf = np.concatenate([c, np.array(oh, np.float32), np.array(props, np.float32)], 1)
    dm = cdist(c, c)
    i, j = np.where((dm < 8.0) & (dm > 0))
    return {'coords': c, 'node_features': nf, 'edges': np.stack([i, j])}


def _chains(pdb):
    s = PDBParser(QUIET=True).get_structure('s', pdb)
    return [c.id for c in next(s.get_models()).get_chains()]


def _bound_lig(pdb, chains):
    s = PDBParser(QUIET=True).get_structure('b', pdb)
    c = []
    for ch in next(s.get_models()):
        if ch.id in chains:
            for r in ch:
                if r.id[0] != ' ' or 'CA' not in r: continue
                c.append(r['CA'].get_coord())
    return np.array(c, np.float32)


# ============================================================
# CAPRI metrics (standard published formulas)
# ============================================================
def _kabsch(m, t):
    n = min(len(m), len(t)); m, t = m[:n], t[:n]
    mc, tc = m.mean(0), t.mean(0)
    U, S, Vt = np.linalg.svd((m - mc).T @ (t - tc))
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    return R, tc - mc @ R.T

def _fnat(pr, pl, nr, nl, t=5.0):
    nc = set(zip(*np.where(cdist(nr, nl) < t)))
    return len(nc & set(zip(*np.where(cdist(pr, pl) < t)))) / len(nc) if nc else 0.0

def _lrmsd(pr, pl, nr, nl):
    n = min(len(pr), len(nr)); m = min(len(pl), len(nl))
    R, t = _kabsch(pr[:n], nr[:n]); return np.sqrt(np.mean(np.sum((pl[:m] @ R.T + t - nl[:m])**2, 1)))

def _irmsd(pr, pl, nr, nl, c=10.0):
    d = cdist(nr, nl); ri, li = np.where(d.min(1)<c)[0], np.where(d.min(0)<c)[0]
    if len(ri)==0 or len(li)==0:
        n=min(len(pl),len(nl)); return np.sqrt(np.mean(np.sum((pl[:n]-nl[:n])**2,1)))
    ni, pi = np.vstack([nr[ri],nl[li]]), np.vstack([pr[ri],pl[li]])
    R,t = _kabsch(pi,ni); a=pi@R.T+t; return np.sqrt(np.mean(np.sum((a-ni)**2,1)))

def _dockq(ir, lr, fn): return (fn + 1/(1+(ir/1.5)**2) + 1/(1+(lr/8.5)**2)) / 3

def _quality(d):
    if d>=0.8: return 'high'
    if d>=0.49: return 'medium'
    if d>=0.23: return 'acceptable'
    return 'incorrect'


# ============================================================
# Write PDB
# ============================================================
def _write(rec_pdb, lig_pdb, pred, rc, lc, out):
    parser = PDBParser(QUIET=True)
    ns = Structure.Structure('pred'); nm = Model.Model(0); ns.add(nm)
    for ch in next(parser.get_structure('r', rec_pdb).get_models()):
        if rc is None or ch.id in rc: nm.add(ch.copy())
    ci = 0
    for ch in next(parser.get_structure('l', lig_pdb).get_models()):
        if lc and ch.id not in lc: continue
        nc = ch.copy()
        for r in nc:
            if r.id[0]!=' ' or 'CA' not in r: continue
            if ci >= len(pred): break
            d = pred[ci] - r['CA'].get_vector().get_array()
            for a in r: a.set_coord(a.get_coord() + d)
            ci += 1
        if nm.has_id(nc.id):
            for x in 'XYZWVU':
                if not nm.has_id(x): nc.id = x; break
        nm.add(nc)
    io = PDBIO(); io.set_structure(ns); io.save(str(out))


# ============================================================
# Main
# ============================================================
def _apply_transform(coords, rotation, translation):
    """Rodrigues' formula: axis-angle to rotation matrix, then apply."""
    angle = torch.norm(rotation)
    if angle < 1e-6:
        R = torch.eye(3, device=coords.device)
    else:
        ax = rotation / angle
        K = torch.zeros(3, 3, device=coords.device)
        K[0,1], K[0,2] = -ax[2], ax[1]
        K[1,0], K[1,2] = ax[2], -ax[0]
        K[2,0], K[2,1] = -ax[1], ax[0]
        R = torch.eye(3, device=coords.device) + torch.sin(angle)*K + (1-torch.cos(angle))*K@K
    return coords @ R.t() + translation.unsqueeze(0)


def _find_model():
    for b in [Path(__file__).parent, Path(__file__).parent / 'experiments']:
        for c in sorted(b.glob('**/dockgraph.pt')) + sorted(b.glob('**/best_model.pt')):
            return c
    return None

def main():
    ap = argparse.ArgumentParser(description='DockGraph: Protein-Protein Docking')
    ap.add_argument('receptor', help='Receptor PDB (unbound)')
    ap.add_argument('ligand', help='Ligand PDB (unbound)')
    ap.add_argument('--bound', default=None, help='Bound complex PDB (for evaluation)')
    ap.add_argument('--rec_chains', nargs='+', default=None)
    ap.add_argument('--lig_chains', nargs='+', default=None)
    ap.add_argument('--checkpoint', default=None, help='Model file path')
    ap.add_argument('-o', '--output', default=None)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rp, lp = Path(args.receptor), Path(args.ligand)
    if not rp.exists(): sys.exit(f"Not found: {rp}")
    if not lp.exists(): sys.exit(f"Not found: {lp}")

    ckpt = Path(args.checkpoint) if args.checkpoint else _find_model()
    if not ckpt or not ckpt.exists(): sys.exit("No model found. Use --checkpoint.")

    print("=" * 70)
    print("DockGraph — Protein-Protein Docking Prediction")
    print("=" * 70)

    # Load model (TorchScript — architecture hidden)
    model = torch.jit.load(str(ckpt), map_location=dev)
    model.eval()

    if args.rec_chains is None: args.rec_chains = _chains(rp)
    if args.lig_chains is None: args.lig_chains = _chains(lp)
    print(f"  Receptor: {rp.name}  chains={args.rec_chains}")
    print(f"  Ligand:   {lp.name}  chains={args.lig_chains}")

    rf = _extract(str(rp), args.rec_chains)
    lf = _extract(str(lp), args.lig_chains)
    rd = {k: torch.tensor(v, dtype=torch.float32 if v.dtype==np.float32 else torch.long).to(dev)
          for k, v in [('node_features',rf['node_features']),('edge_index',rf['edges']),('coords',rf['coords'])]}
    ld = {k: torch.tensor(v, dtype=torch.float32 if v.dtype==np.float32 else torch.long).to(dev)
          for k, v in [('node_features',lf['node_features']),('edge_index',lf['edges']),('coords',lf['coords'])]}

    print("\nPredicting...")
    with torch.no_grad():
        rot, trans, conf = model(rd, ld)
        pc = _apply_transform(ld['coords'], rot, trans)

    pn = pc.cpu().numpy()
    d = np.linalg.norm(pn.mean(0) - lf['coords'].mean(0))
    print(f"  Confidence: {conf.item():.4f}")
    print(f"  Displacement: {d:.2f} A")

    od = Path(__file__).parent / 'predicted_complex'
    od.mkdir(parents=True, exist_ok=True)
    op = Path(args.output) if args.output else od / 'predicted_complex.pdb'
    op.parent.mkdir(parents=True, exist_ok=True)
    _write(str(rp), str(lp), pn, args.rec_chains, args.lig_chains, op)
    print(f"  Output: {op}")

    if args.bound:
        bp = Path(args.bound)
        if not bp.exists(): print(f"  Not found: {bp}"); return
        print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
        nl = _bound_lig(str(bp), args.lig_chains)
        n = min(len(pn), len(nl))
        if n == 0: print("  No ligand residues found."); return
        pr, pl, nr, nli = rf['coords'], pn[:n], rf['coords'], nl[:n]
        fn = _fnat(pr,pl,nr,nli); lr = _lrmsd(pr,pl,nr,nli)
        ir = _irmsd(pr,pl,nr,nli); dq = _dockq(ir,lr,fn)
        print(f"  DockQ:   {dq:.4f}  ({_quality(dq)})")
        print(f"  fnat:    {fn:.4f}")
        print(f"  I-RMSD:  {ir:.4f} A")
        print(f"  L-RMSD:  {lr:.4f} A")
        print(f"  Acceptable+: {'YES' if dq>=0.23 else 'NO'}")

    print(f"\n{'='*70}\nDone.\n{'='*70}")

if __name__ == "__main__": main()