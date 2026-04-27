# CASP15 Evaluation Pipeline for DockGraph

Evaluate DockGraph docking refinement on CASP15 blind targets, starting from sequence-only inputs predicted with AlphaFold3.

## Target: H1141 (CASP15 T206)

| Property | Value |
|----------|-------|
| **CASP Target** | T206 / H1141 |
| **Protein** | CNPase-Nb7e (mouse CNPase catalytic domain + alpaca nanobody 7E) |
| **Organism** | Mouse (*Mus musculus*) / Alpaca (*Vicugna pacos*) |
| **Residues** | 346 (219 + 127) |
| **PDB ID** | [9ERU](https://www.rcsb.org/structure/9ERU) |
| **Method** | X-ray crystallography |
| **Complex Type** | Nanobody–antigen |
| **Chain A** | CNPase catalytic domain (receptor/antigen) |
| **Chain B** | Nanobody 7E (ligand) |

### Biological Context

CNPase (2',3'-cyclic nucleotide 3'-phosphodiesterase) is an abundant myelin sheath enzyme. Nanobody 7E binds near the CNPase active site — its CDR1/CDR2 loops alter the conformation of a flexible loop and Arg224, while the elongated CDR3 forms β-sheet interactions with the C-terminal region. This makes it a challenging docking target due to induced-fit conformational changes at the interface.

## Pipeline Overview

```
Sequence (FASTA)
    │
    ▼
AlphaFold3 Server ──────► Predicted complex structure (mmCIF)
    │
    ▼
Parse & Split Chains ───► receptor.pdb + ligand.pdb  
    │
    ▼
DockGraph Refinement ───► refined_complex.pdb (rotation + translation)
    │
    ▼
Evaluate vs PDB 9ERU ──► DockQ, I-RMSD, L-RMSD, fnat, CAPRI class
```

## Quick Start

### Step 1: Download Ground Truth
```bash
python download_ground_truth.py --target H1141 --output_dir ground_truth/
```

### Step 2: Run AlphaFold3 Prediction

**Option A: AlphaFold Server (recommended for convenience)**

1. Go to [alphafoldserver.com](https://alphafoldserver.com)
2. Upload `alphafold_inputs/H1141_T206_alphafold_server.json`  
   OR paste the two sequences from `H1141_T206.fasta`
3. Submit and download results

**Option B: Local AlphaFold3**
```bash
python run_alphafold.py \
    --json_path alphafold_inputs/H1141_T206_alphafold3.json \
    --output_dir alphafold_outputs/H1141/ \
    --model_dir /path/to/af3_weights/
```

**Option C: ColabFold** (AF2-Multimer, free)
```bash
# Upload H1141_T206.fasta to ColabFold notebook
# Set model_type=alphafold2_multimer_v3
```

### Step 3: Run DockGraph Refinement
```bash
python predict_casp15.py \
    --af3_output_dir alphafold_outputs/H1141/ \
    --model_path /path/to/best_model.pt \
    --output_dir predictions/H1141/
```

### Step 4: Evaluate
```bash
python evaluate_casp15.py \
    --prediction predictions/H1141/refined_complex.pdb \
    --ground_truth ground_truth/H1141/9ERU.cif \
    --receptor_chain A --ligand_chain B \
    --output_dir results/H1141/
```

### Full Pipeline (single command)
```bash
export DOCKGRAPH_MODEL=/path/to/best_model.pt
bash run_casp15_pipeline.sh --af3-dir alphafold_outputs/H1141/
```

## Evaluation Metrics

### DockQ Score
Combined quality metric ∈ [0, 1] (Basu & Wallner, 2016):

```
DockQ = (fnat + 1/(1+(I-RMSD/1.5)²) + 1/(1+(L-RMSD/8.5)²)) / 3
```

| DockQ Range | Classification |
|-------------|---------------|
| ≥ 0.80 | High |
| (0.49, 0.80) | Medium |
| [0.23, 0.49] | Acceptable |
| < 0.23 | Incorrect |

### CAPRI Classification

| Quality | fnat | L-RMSD (Å) | I-RMSD (Å) |
|---------|------|-------------|-------------|
| **High** | ≥ 0.5 | ≤ 1.0 OR | ≤ 1.0 |
| **Medium** | ≥ 0.3 | ≤ 5.0 OR | ≤ 2.0 |
| **Acceptable** | ≥ 0.1 | ≤ 10.0 OR | ≤ 4.0 |
| **Incorrect** | otherwise | — | — |

### Individual Metrics

- **fnat**: Fraction of native inter-chain contacts recovered (contact threshold: 5.0 Å)
- **I-RMSD**: Interface RMSD — backbone RMSD of interface residues after superposition
- **L-RMSD**: Ligand RMSD — backbone RMSD of ligand after receptor superposition
- **Contact threshold**: 5.0 Å (heavy atoms, standard CAPRI)
- **Interface threshold**: 10.0 Å (for identifying interface residues)

## AlphaFold3 JSON Input Formats

### Server Format (`alphafold_server.json`)
```json
[{
    "name": "H1141_T206_CNPase_Nb7e",
    "modelSeeds": [42, 123, 2024, 7777, 9999],
    "sequences": [
        {"proteinChain": {"sequence": "GLEK...", "count": 1}},
        {"proteinChain": {"sequence": "EVQL...", "count": 1}}
    ]
}]
```

### Local Format (`alphafold3.json`)
```json
{
    "name": "H1141_T206_CNPase_Nb7e",
    "modelSeeds": [42, 123, 2024, 7777, 9999],
    "sequences": [
        {"protein": {"id": "A", "sequence": "GLEK..."}},
        {"protein": {"id": "B", "sequence": "EVQL..."}}
    ],
    "dialect": "alphafold3",
    "version": 4
}
```

## Comparison with AlphaRED (Harmalkar et al.)

AlphaRED evaluated on the same CASP15 nanobody-antigen targets:
- Obtained DockQ > 0.23 for all five targets (T205–T209)
- Medium-quality models (DockQ > 0.49) for T205, T207, T208
- Used ColabFold for initial structures, then Rosetta-based ReplicaDock 2.0

DockGraph aims to achieve comparable or better refinement using end-to-end deep learning, without the multi-stage physics-based pipeline.

## File Structure

```
casp15_evaluation/
├── README.md
├── run_casp15_pipeline.sh          # Master pipeline script
├── download_ground_truth.py        # Download PDB 9ERU
├── predict_casp15.py               # AF3 → DockGraph inference
├── evaluate_casp15.py              # DockQ/CAPRI evaluation
├── alphafold_inputs/
│   ├── H1141_T206.fasta            # Input sequences
│   ├── H1141_T206_alphafold_server.json  # AF3 Server format
│   └── H1141_T206_alphafold3.json  # AF3 local format
├── ground_truth/                   # Downloaded crystal structures
│   └── H1141/
│       ├── 9ERU.cif
│       ├── chains/
│       └── metadata.json
├── alphafold_outputs/              # AF3 predictions (user provides)
├── predictions/                    # DockGraph refined structures
└── results/                        # Evaluation metrics
```

## Requirements

```
biopython>=1.80
numpy>=1.21
torch>=1.12  # For DockGraph inference
```

## References

1. Basu S, Wallner B. DockQ: A Quality Measure for Protein-Protein Docking Models. *PLoS ONE* 11(8), 2016.
2. Harmalkar A, et al. AlphaRED: AlphaFold-driven REfinement for accurate Docking. *eLife* 2025.
3. Kursula P, et al. Nanobodies against the myelin enzyme CNPase. *J Neurochem* 2024.
4. Abramson J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 2024.
5. Lensink MF, et al. Prediction of protein assemblies. *Proteins* 2023 (CASP15-CAPRI).
