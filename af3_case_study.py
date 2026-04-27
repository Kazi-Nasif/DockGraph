#!/usr/bin/env python3
"""
AF3 → DockGraph Refinement Case Study (Simplified)
====================================================
Uses the existing predict.py for DockGraph inference.

Steps:
  1. Parse AF3 mmCIF → split into receptor.pdb + ligand.pdb
  2. Evaluate raw AF3 prediction against ground truth (DockQ)
  3. Call predict.py for DockGraph refinement
  4. Evaluate refined prediction against ground truth (DockQ)
  5. Print before/after comparison

Usage:
    cd ~/Protein\ Docking/DockGraph
    python af3_case_study.py
"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select, Superimposer
from scipy.spatial.distance import cdist

# =====================================================================
# Configuration — edit these paths if needed
# =====================================================================
PROJECT_ROOT = Path(__file__).parent
AF3_DIR = PROJECT_ROOT / "af3_case_study" / "af3_output"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
OUTPUT_DIR = PROJECT_ROOT / "af3_case_study" / "results"
PREDICT_PY = PROJECT_ROOT / "scripts" / "step7_predict.py"

# Find checkpoint automatically
ckpt_candidates = sorted(PROJECT_ROOT.glob("experiments/*/best_model.pt"))
CHECKPOINT = ckpt_candidates[-1] if ckpt_candidates else None

# Target: 2DD8 (m396 antibody vs SARS-CoV Spike RBD)
TARGET = "2DD8"
# AF3 Server chain order: A=Heavy, B=Light, C=Spike
# DB5.5 chain order: H=Heavy, L=Light, S=Spike
AF3_REC_CHAINS = ["C"]       # Spike (receptor in DB5.5 convention)
AF3_LIG_CHAINS = ["A", "B"]  # Antibody H+L (ligand in DB5.5 convention)
NATIVE_REC_CHAINS = ["S"]
NATIVE_LIG_CHAINS = ["H", "L"]


# =====================================================================
# Step 1: Parse AF3 output → receptor.pdb + ligand.pdb
# =====================================================================
def parse_af3():
    """Parse AF3 mmCIF, split into receptor and ligand PDBs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    cif_files = sorted(AF3_DIR.glob("*_model_0.cif"))
    if not cif_files:
        cif_files = sorted(AF3_DIR.glob("*.cif"))
    if not cif_files:
        sys.exit(f"No .cif files in {AF3_DIR}")
    
    best_cif = cif_files[0]
    print(f"  AF3 model: {best_cif.name}")
    
    # Parse confidence
    conf_files = sorted(AF3_DIR.glob("*_summary_confidences_0.json"))
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        print(f"  pTM={conf.get('ptm',0):.3f}, ipTM={conf.get('iptm',0):.3f}, "
              f"ranking={conf.get('ranking_score',0):.3f}")
    
    # Parse structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("af3", str(best_cif))
    
    class ChainSelect(Select):
        def __init__(self, chains):
            self.chains = chains
        def accept_chain(self, chain):
            return chain.id in self.chains
        def accept_residue(self, res):
            return res.id[0] == ' '
    
    io = PDBIO()
    
    # Save receptor (spike = chain C)
    rec_pdb = OUTPUT_DIR / "af3_receptor.pdb"
    io.set_structure(structure)
    io.save(str(rec_pdb), ChainSelect(AF3_REC_CHAINS))
    
    # Save ligand (antibody = chains A, B)
    lig_pdb = OUTPUT_DIR / "af3_ligand.pdb"
    io.set_structure(structure)
    io.save(str(lig_pdb), ChainSelect(AF3_LIG_CHAINS))
    
    # Save full complex
    complex_pdb = OUTPUT_DIR / "af3_complex.pdb"
    io.set_structure(structure)
    io.save(str(complex_pdb), ChainSelect(AF3_REC_CHAINS + AF3_LIG_CHAINS))
    
    # Count residues
    model = next(structure.get_models())
    n_rec = sum(1 for ch in model if ch.id in AF3_REC_CHAINS
                for r in ch if r.id[0] == ' ')
    n_lig = sum(1 for ch in model if ch.id in AF3_LIG_CHAINS
                for r in ch if r.id[0] == ' ')
    print(f"  Receptor (spike, chain C): {n_rec} residues")
    print(f"  Ligand (antibody, chains A+B): {n_lig} residues")
    
    return rec_pdb, lig_pdb, complex_pdb


# =====================================================================
# Step 2: DockQ evaluation (chain-agnostic by residue index)
# =====================================================================
def get_ca_coords_by_group(pdb_path, chain_group):
    """Get CA coordinates for a group of chains, ordered by chain then residue."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    model = next(struct.get_models())
    coords = []
    for ch in model:
        if ch.id in chain_group:
            for res in ch:
                if res.id[0] == ' ' and 'CA' in res:
                    coords.append(res['CA'].get_vector().get_array())
    return np.array(coords, dtype=np.float64)


def get_bb_coords_by_group(pdb_path, chain_group):
    """Get backbone (N,CA,C,O) coordinates for chain group."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("s", pdb_path)
    model = next(struct.get_models())
    coords = []
    for ch in model:
        if ch.id in chain_group:
            for res in ch:
                if res.id[0] != ' ':
                    continue
                for aname in ["N", "CA", "C", "O"]:
                    if aname in res:
                        coords.append(res[aname].get_vector().get_array())
    return np.array(coords, dtype=np.float64)


def compute_fnat(pred_pdb, pred_rec_chains, pred_lig_chains,
                 native_pdb, native_rec_chains, native_lig_chains,
                 threshold=5.0):
    """
    Compute fnat by matching contacts via residue INDEX within each chain group.
    This handles different chain IDs between prediction and native.
    """
    parser = PDBParser(QUIET=True)
    
    def get_contacts(pdb_path, rec_chains, lig_chains):
        struct = parser.get_structure("s", pdb_path)
        model = next(struct.get_models())
        
        # Collect residues with sequential index
        rec_residues = []
        for ch in model:
            if ch.id in rec_chains:
                for res in ch:
                    if res.id[0] == ' ':
                        atoms = [a for a in res if a.element != 'H']
                        if atoms:
                            rec_residues.append((len(rec_residues), atoms))
        
        lig_residues = []
        for ch in model:
            if ch.id in lig_chains:
                for res in ch:
                    if res.id[0] == ' ':
                        atoms = [a for a in res if a.element != 'H']
                        if atoms:
                            lig_residues.append((len(lig_residues), atoms))
        
        contacts = set()
        for ri, (ridx, ratoms) in enumerate(rec_residues):
            for li, (lidx, latoms) in enumerate(lig_residues):
                found = False
                for a1 in ratoms:
                    for a2 in latoms:
                        if (a1 - a2) < threshold:
                            contacts.add((ridx, lidx))
                            found = True
                            break
                    if found:
                        break
        return contacts
    
    native_contacts = get_contacts(native_pdb, native_rec_chains, native_lig_chains)
    pred_contacts = get_contacts(pred_pdb, pred_rec_chains, pred_lig_chains)
    
    if len(native_contacts) == 0:
        return 0.0, 0, 0
    
    recovered = native_contacts & pred_contacts
    fnat = len(recovered) / len(native_contacts)
    return fnat, len(native_contacts), len(pred_contacts)


def kabsch_rmsd(coords_pred, coords_native):
    """RMSD after optimal superposition (Kabsch algorithm)."""
    n = min(len(coords_pred), len(coords_native))
    if n == 0:
        return 999.0
    p = coords_pred[:n].copy()
    q = coords_native[:n].copy()
    
    cp, cq = p.mean(0), q.mean(0)
    p -= cp
    q -= cq
    
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    
    p_rot = (R @ p.T).T
    return np.sqrt(np.mean(np.sum((p_rot - q)**2, axis=1)))


def compute_lrmsd(pred_pdb, pred_rec_chains, pred_lig_chains,
                  native_pdb, native_rec_chains, native_lig_chains):
    """L-RMSD: superpose receptors, then measure ligand displacement."""
    rec_pred = get_bb_coords_by_group(pred_pdb, pred_rec_chains)
    rec_native = get_bb_coords_by_group(native_pdb, native_rec_chains)
    lig_pred = get_bb_coords_by_group(pred_pdb, pred_lig_chains)
    lig_native = get_bb_coords_by_group(native_pdb, native_lig_chains)
    
    n_rec = min(len(rec_pred), len(rec_native))
    n_lig = min(len(lig_pred), len(lig_native))
    if n_rec == 0 or n_lig == 0:
        return 999.0
    
    rp, rn = rec_pred[:n_rec], rec_native[:n_rec]
    lp, ln = lig_pred[:n_lig], lig_native[:n_lig]
    
    # Kabsch on receptor
    cp, cn = rp.mean(0), rn.mean(0)
    H = (rp - cp).T @ (rn - cn)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    
    # Apply to ligand
    lp_transformed = (R @ (lp - cp).T).T + cn
    return np.sqrt(np.mean(np.sum((lp_transformed - ln)**2, axis=1)))


def compute_irmsd(pred_pdb, pred_rec_chains, pred_lig_chains,
                  native_pdb, native_rec_chains, native_lig_chains,
                  interface_threshold=10.0):
    """I-RMSD: RMSD of interface backbone atoms after optimal superposition."""
    parser = PDBParser(QUIET=True)
    
    def get_interface_bb(pdb_path, rec_chains, lig_chains):
        struct = parser.get_structure("s", pdb_path)
        model = next(struct.get_models())
        
        # Find interface residues (CA within threshold of partner)
        rec_cas, lig_cas = [], []
        rec_res_list, lig_res_list = [], []
        
        for ch in model:
            if ch.id in rec_chains:
                for res in ch:
                    if res.id[0] == ' ' and 'CA' in res:
                        rec_cas.append(res['CA'].get_vector().get_array())
                        rec_res_list.append((ch.id, res.id))
            elif ch.id in lig_chains:
                for res in ch:
                    if res.id[0] == ' ' and 'CA' in res:
                        lig_cas.append(res['CA'].get_vector().get_array())
                        lig_res_list.append((ch.id, res.id))
        
        if not rec_cas or not lig_cas:
            return np.array([])
        
        rec_ca = np.array(rec_cas)
        lig_ca = np.array(lig_cas)
        dist = cdist(rec_ca, lig_ca)
        
        interface_indices = set()
        for i in range(len(rec_ca)):
            if dist[i].min() < interface_threshold:
                interface_indices.add(('R', i))
        for j in range(len(lig_ca)):
            if dist[:, j].min() < interface_threshold:
                interface_indices.add(('L', j))
        
        # Get backbone coords of interface residues
        bb_coords = []
        rec_idx, lig_idx = 0, 0
        for ch in model:
            if ch.id in rec_chains:
                for res in ch:
                    if res.id[0] == ' ' and 'CA' in res:
                        if ('R', rec_idx) in interface_indices:
                            for aname in ["N", "CA", "C", "O"]:
                                if aname in res:
                                    bb_coords.append(res[aname].get_vector().get_array())
                        rec_idx += 1
            elif ch.id in lig_chains:
                for res in ch:
                    if res.id[0] == ' ' and 'CA' in res:
                        if ('L', lig_idx) in interface_indices:
                            for aname in ["N", "CA", "C", "O"]:
                                if aname in res:
                                    bb_coords.append(res[aname].get_vector().get_array())
                        lig_idx += 1
        
        return np.array(bb_coords, dtype=np.float64) if bb_coords else np.array([])
    
    native_ibb = get_interface_bb(native_pdb, native_rec_chains, native_lig_chains)
    pred_ibb = get_interface_bb(pred_pdb, pred_rec_chains, pred_lig_chains)
    
    if len(native_ibb) == 0 or len(pred_ibb) == 0:
        return 999.0
    
    return kabsch_rmsd(pred_ibb, native_ibb)


def full_evaluate(pred_pdb, native_pdb, pred_rec, pred_lig, nat_rec, nat_lig):
    """Run full DockQ evaluation."""
    fnat, n_nat, n_pred = compute_fnat(pred_pdb, pred_rec, pred_lig,
                                        native_pdb, nat_rec, nat_lig)
    irmsd = compute_irmsd(pred_pdb, pred_rec, pred_lig,
                          native_pdb, nat_rec, nat_lig)
    lrmsd = compute_lrmsd(pred_pdb, pred_rec, pred_lig,
                          native_pdb, nat_rec, nat_lig)
    
    dockq = (fnat + 1.0/(1.0+(irmsd/1.5)**2) + 1.0/(1.0+(lrmsd/8.5)**2)) / 3.0
    
    if dockq >= 0.80:
        quality = "High"
    elif dockq >= 0.49:
        quality = "Medium"
    elif dockq >= 0.23:
        quality = "Acceptable"
    else:
        quality = "Incorrect"
    
    return {
        "dockq": dockq, "fnat": fnat, "irmsd": irmsd, "lrmsd": lrmsd,
        "quality": quality, "n_native_contacts": n_nat, "n_pred_contacts": n_pred
    }


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("AF3 → DockGraph Refinement Case Study")
    print("Target: 2DD8 — m396 antibody vs SARS-CoV Spike RBD")
    print("=" * 70)
    
    # Ground truth
    gt_bound = BENCHMARK_DIR / "medium_targets" / TARGET / f"{TARGET}_b.pdb"
    gt_rec_u = BENCHMARK_DIR / "medium_targets" / TARGET / f"{TARGET}_r_u.pdb"
    gt_lig_u = BENCHMARK_DIR / "medium_targets" / TARGET / f"{TARGET}_l_u.pdb"
    
    if not gt_bound.exists():
        sys.exit(f"Ground truth not found: {gt_bound}")
    
    # ── Step 1: Parse AF3 ──
    print("\n─── Step 1: Parsing AlphaFold 3 output ───")
    rec_pdb, lig_pdb, complex_pdb = parse_af3()
    
    # ── Step 2: Evaluate raw AF3 prediction ──
    print("\n─── Step 2: Evaluating raw AF3 prediction ───")
    af3_eval = full_evaluate(
        str(complex_pdb), str(gt_bound),
        AF3_REC_CHAINS, AF3_LIG_CHAINS,
        NATIVE_REC_CHAINS, NATIVE_LIG_CHAINS
    )
    print(f"  DockQ:  {af3_eval['dockq']:.4f}  ({af3_eval['quality']})")
    print(f"  fnat:   {af3_eval['fnat']:.4f}  ({af3_eval['n_native_contacts']} native contacts)")
    print(f"  I-RMSD: {af3_eval['irmsd']:.2f} Å")
    print(f"  L-RMSD: {af3_eval['lrmsd']:.2f} Å")
    
    # ── Step 3: Run DockGraph via predict.py ──
    print("\n─── Step 3: Running DockGraph refinement (via predict.py) ───")
    
    if CHECKPOINT is None:
        sys.exit("No checkpoint found in experiments/*/best_model.pt")
    
    refined_pdb = OUTPUT_DIR / "dockgraph_refined.pdb"
    
    cmd = [
        sys.executable, str(PREDICT_PY),
        str(rec_pdb), str(lig_pdb),
        "--checkpoint", str(CHECKPOINT),
        "-o", str(refined_pdb),
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("  STDERR:", result.stderr)
        sys.exit("DockGraph prediction failed")
    
    # ── Step 4: Evaluate refined prediction ──
    print("─── Step 4: Evaluating DockGraph-refined prediction ───")
    
    if not refined_pdb.exists():
        # predict.py might save to default location
        alt = PROJECT_ROOT / "predicted_complex" / "predicted_complex.pdb"
        if alt.exists():
            refined_pdb = alt
        else:
            sys.exit(f"Refined PDB not found at {refined_pdb} or {alt}")
    
    # Determine chain IDs in refined PDB
    p = PDBParser(QUIET=True)
    s = p.get_structure("ref", str(refined_pdb))
    ref_chains = [ch.id for ch in next(s.get_models())]
    print(f"  Refined PDB chains: {ref_chains}")
    
    # The refined PDB preserves AF3 chain IDs: C=receptor, A+B=ligand
    refined_eval = full_evaluate(
        str(refined_pdb), str(gt_bound),
        AF3_REC_CHAINS, AF3_LIG_CHAINS,
        NATIVE_REC_CHAINS, NATIVE_LIG_CHAINS
    )
    print(f"  DockQ:  {refined_eval['dockq']:.4f}  ({refined_eval['quality']})")
    print(f"  fnat:   {refined_eval['fnat']:.4f}")
    print(f"  I-RMSD: {refined_eval['irmsd']:.2f} Å")
    print(f"  L-RMSD: {refined_eval['lrmsd']:.2f} Å")
    
    # ── Step 5: Also run DockGraph on DB5.5 unbound (reference) ──
    print("\n─── Step 5: DockGraph on DB5.5 unbound (reference) ───")
    if gt_rec_u.exists() and gt_lig_u.exists():
        unbound_pdb = OUTPUT_DIR / "dockgraph_unbound.pdb"
        cmd2 = [
            sys.executable, str(PREDICT_PY),
            str(gt_rec_u), str(gt_lig_u),
            "--bound", str(gt_bound),
            "--lig_chains", "H", "L",
            "--checkpoint", str(CHECKPOINT),
            "-o", str(unbound_pdb),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        print(result2.stdout)
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Target: {TARGET} — m396 antibody vs SARS-CoV Spike RBD")
    print(f"  Category: Antibody-Antigen, Medium difficulty\n")
    
    dq = af3_eval['dockq']
    dq2 = refined_eval['dockq']
    print(f"  {'Metric':<12} {'AF3 raw':>10} {'+ DockGraph':>12} {'Change':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*10}")
    print(f"  {'DockQ':<12} {dq:>10.4f} {dq2:>12.4f} {dq2-dq:>+10.4f}")
    print(f"  {'fnat':<12} {af3_eval['fnat']:>10.4f} {refined_eval['fnat']:>12.4f} {refined_eval['fnat']-af3_eval['fnat']:>+10.4f}")
    print(f"  {'I-RMSD':<12} {af3_eval['irmsd']:>10.2f} {refined_eval['irmsd']:>12.2f} {refined_eval['irmsd']-af3_eval['irmsd']:>+10.2f}")
    print(f"  {'L-RMSD':<12} {af3_eval['lrmsd']:>10.2f} {refined_eval['lrmsd']:>12.2f} {refined_eval['lrmsd']-af3_eval['lrmsd']:>+10.2f}")
    print(f"  {'Quality':<12} {af3_eval['quality']:>10} {refined_eval['quality']:>12}")
    
    # Save results
    results = {
        "target": TARGET,
        "af3_raw": af3_eval,
        "dockgraph_refined": refined_eval,
    }
    results_json = OUTPUT_DIR / "case_study_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()