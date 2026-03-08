# DockGraph

A graph neural network for protein-protein docking refinement.

## Results

Evaluated on 253 targets from Docking Benchmark 5.5 using CAPRI metrics:

| Metric | Value |
|--------|-------|
| Success rate (DockQ ≥ 0.23) | **98.0%** |
| High quality (DockQ ≥ 0.8) | 44.3% |
| Medium quality (DockQ ≥ 0.49) | 87.7% |
| Mean DockQ | 0.728 |
| Mean I-RMSD | 1.28 Å |
| Mean fnat | 0.545 |

| Difficulty | Targets | Success Rate | Mean DockQ | Mean I-RMSD |
|------------|---------|-------------|------------|-------------|
| Rigid | 159 | 98.1% | 0.768 | 0.98 Å |
| Medium | 59 | 100.0% | 0.696 | 1.32 Å |
| Difficult | 35 | 94.3% | 0.602 | 2.58 Å |

## Installation
```bash
git clone https://github.com/Kazi-Nasif/DockGraph.git
cd DockGraph
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU (optional).

## Data

Docking Benchmark 5.5 is included under `data/benchmark/`. Original source: [Graylab/AlphaRED](https://github.com/Graylab/AlphaRED/tree/main/benchmark).

## Usage
```bash
python predict.py receptor.pdb ligand.pdb [options]
```

| Flag | Description |
|------|-------------|
| `--bound` | Bound complex PDB for evaluation |
| `--lig_chains` | Ligand chain IDs (default: all) |
| `--rec_chains` | Receptor chain IDs (default: all) |
| `-o` | Output PDB path |
| `--checkpoint` | Model file path |

## Example
```bash
python predict.py data/benchmark/rigid_targets/1A2K/1A2K_r_u.pdb \
                  data/benchmark/rigid_targets/1A2K/1A2K_l_u.pdb \
                  --bound data/benchmark/rigid_targets/1A2K/1A2K_b.pdb \
                  --lig_chains C
```
```
======================================================================
DockGraph — Protein-Protein Docking Prediction
======================================================================
  Receptor: 1A2K_r_u.pdb  chains=['A', 'B']
  Ligand:   1A2K_l_u.pdb  chains=['C']

  Confidence: 0.9897
  Displacement: 0.09 A

EVALUATION
  DockQ:   0.9792  (high)
  fnat:    1.0000
  I-RMSD:  0.3247 A
  L-RMSD:  1.1396 A
  Acceptable+: YES
```

Predicted complex is saved to `predicted_complex/predicted_complex.pdb`.

## License

MIT
