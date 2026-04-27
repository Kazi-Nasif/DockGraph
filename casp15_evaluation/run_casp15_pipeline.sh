#!/bin/bash
# =============================================================================
# run_casp15_pipeline.sh
# =============================================================================
# Master pipeline for DockGraph evaluation on CASP15 targets.
#
# Pipeline:
#   1. Download ground truth crystal structure (PDB 9ERU)
#   2. [MANUAL] Run AlphaFold3 Server with the JSON input to get predictions
#   3. Parse AF3 output and run DockGraph refinement
#   4. Evaluate against ground truth with DockQ/CAPRI metrics
#
# Usage:
#   bash run_casp15_pipeline.sh [--skip-download] [--af3-dir <path>]
# =============================================================================

set -e

# Configuration
TARGET="H1141"
CASP_TARGET="T206"
MODEL_PATH="${DOCKGRAPH_MODEL:-/path/to/DockGraph/experiments/20251126_185619_full_training/best_model.pt}"
DEVICE="${DEVICE:-cuda}"

# Directories
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
GT_DIR="${BASE_DIR}/ground_truth"
PRED_DIR="${BASE_DIR}/predictions/${TARGET}"
RESULTS_DIR="${BASE_DIR}/results/${TARGET}"
AF3_INPUT_DIR="${BASE_DIR}/alphafold_inputs"

echo "============================================================"
echo "DockGraph CASP15 Evaluation Pipeline"
echo "============================================================"
echo "Target: ${TARGET} (${CASP_TARGET})"
echo "Model:  ${MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "============================================================"

# -------------------------------------------------------
# Step 1: Download Ground Truth
# -------------------------------------------------------
echo ""
echo "[Step 1/4] Downloading ground truth structure..."
if [ "$1" != "--skip-download" ]; then
    python "${BASE_DIR}/download_ground_truth.py" \
        --target "${TARGET}" \
        --output_dir "${GT_DIR}"
else
    echo "  Skipped (--skip-download)"
fi

# -------------------------------------------------------
# Step 2: AlphaFold3 Prediction (MANUAL)
# -------------------------------------------------------
echo ""
echo "[Step 2/4] AlphaFold3 Structure Prediction"
echo "============================================================"
echo "  This step requires MANUAL intervention."
echo ""
echo "  Option A: AlphaFold3 Server (https://alphafoldserver.com)"
echo "    1. Go to https://alphafoldserver.com"
echo "    2. Upload the JSON file:"
echo "       ${AF3_INPUT_DIR}/H1141_T206_alphafold_server.json"
echo "       OR paste the FASTA from:"
echo "       ${AF3_INPUT_DIR}/H1141_T206.fasta"
echo "    3. Submit job (generates 5 models with different seeds)"
echo "    4. Download results and extract to a directory"
echo ""
echo "  Option B: Local AlphaFold3 (if installed on server)"
echo "    python run_alphafold.py \\"
echo "      --json_path ${AF3_INPUT_DIR}/H1141_T206_alphafold3.json \\"
echo "      --output_dir alphafold_outputs/${TARGET}/ \\"
echo "      --model_dir /path/to/af3_weights/"
echo ""
echo "  Option C: ColabFold (free, no GPU required)"
echo "    Upload ${AF3_INPUT_DIR}/H1141_T206.fasta to"
echo "    https://colab.research.google.com/github/sokrypton/ColabFold"
echo "    Set model_type=alphafold2_multimer_v3"
echo ""

# Check if AF3 output exists
AF3_DIR="${2:-alphafold_outputs/${TARGET}}"
if [ -d "${AF3_DIR}" ]; then
    echo "  Found AF3 output at: ${AF3_DIR}"
else
    echo "  Waiting for AF3 output at: ${AF3_DIR}"
    echo "  Re-run this script with: --af3-dir <path_to_af3_output>"
    echo ""
    echo "  To continue with evaluation only (if you already have"
    echo "  a predicted complex PDB), run directly:"
    echo ""
    echo "    python evaluate_casp15.py \\"
    echo "      --prediction <your_prediction.pdb> \\"
    echo "      --ground_truth ${GT_DIR}/${TARGET}/9ERU.cif \\"
    echo "      --receptor_chain A --ligand_chain B \\"
    echo "      --output_dir ${RESULTS_DIR}"
    echo ""
    exit 0
fi

# -------------------------------------------------------
# Step 3: DockGraph Refinement
# -------------------------------------------------------
echo ""
echo "[Step 3/4] Running DockGraph refinement..."
python "${BASE_DIR}/predict_casp15.py" \
    --af3_output_dir "${AF3_DIR}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${PRED_DIR}" \
    --device "${DEVICE}"

# -------------------------------------------------------
# Step 4: Evaluation
# -------------------------------------------------------
echo ""
echo "[Step 4/4] Evaluating against ground truth..."

# Determine ground truth file
GT_FILE="${GT_DIR}/${TARGET}/9ERU.pdb"
if [ ! -f "${GT_FILE}" ]; then
    GT_FILE="${GT_DIR}/${TARGET}/9ERU.cif"
fi

python "${BASE_DIR}/evaluate_casp15.py" \
    --prediction "${PRED_DIR}/refined_complex.pdb" \
    --ground_truth "${GT_FILE}" \
    --receptor_chain A \
    --ligand_chain B \
    --output_dir "${RESULTS_DIR}"

# Also evaluate the raw AF3 prediction (without DockGraph refinement)
echo ""
echo "[Bonus] Evaluating raw AlphaFold3 prediction (no refinement)..."
python "${BASE_DIR}/evaluate_casp15.py" \
    --prediction "${PRED_DIR}/parsed/af3_complex.pdb" \
    --ground_truth "${GT_FILE}" \
    --receptor_chain A \
    --ligand_chain B \
    --output_dir "${RESULTS_DIR}/af3_baseline"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Results:"
echo "  DockGraph: ${RESULTS_DIR}/evaluation_results.json"
echo "  AF3 base:  ${RESULTS_DIR}/af3_baseline/evaluation_results.json"
echo "============================================================"
