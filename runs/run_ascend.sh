#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end script for Ascend NPU training.
# Assumes PyTorch + torch_npu + CANN are already installed and working.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Basic runtime env (override from shell if needed)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${REPO_DIR}/out}"

# Optional network/mirror settings (uncomment or export before running)
# export HTTP_PROXY="http://host:port"
# export HTTPS_PROXY="http://host:port"
# export HF_ENDPOINT="https://hf-mirror.example.com"
# export PIP_INDEX_URL="https://pypi.org/simple"

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "INFO: SWANLAB_API_KEY is not set. SwanLab will use an existing local session if available."
fi

# Install Python dependencies needed by this script. Use PIP_INDEX_URL if provided.
PIP_CMD=(python -m pip install wandb swanlab "rustbpe>=0.1.0")
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  PIP_CMD+=(-i "${PIP_INDEX_URL}")
fi
"${PIP_CMD[@]}"

# Optional prep stages
RUN_REPORT_RESET="${RUN_REPORT_RESET:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-1}"
TRAIN_TOKENIZER="${TRAIN_TOKENIZER:-1}"

if [[ "${RUN_REPORT_RESET}" == "1" ]]; then
  python -m nanochat.report reset
fi

if [[ "${DOWNLOAD_DATASETS}" == "1" ]]; then
  python -m nanochat.dataset -n 8     # mini
  python -m nanochat.dataset -n 370   # medium
  python -m nanochat.dataset -n 1000  # large
  python -m nanochat.dataset -n 1822  # full
fi

if [[ "${TRAIN_TOKENIZER}" == "1" ]]; then
  python -m scripts.tok_train
  python -m scripts.tok_eval
fi

# Training config (override via environment variables)
nnpu="${NNPU:-4}"
depth="${DEPTH:-26}"
data_ratio="${TARGET_PARAM_DATA_RATIO:-8.25}"
batch_size="${DEVICE_BATCH_SIZE:-16}"
eval_batch_size="${EVAL_DEVICE_BATCH_SIZE:-32}"
window_pattern="${WINDOW_PATTERN:-L}"   # L|S|SL
run_name="${RUN_NAME:-d${depth}-npu${nnpu}-b${batch_size}-w${window_pattern}}"
sft_name="${SFT_RUN_NAME:-sft-d${depth}-npu${nnpu}-b${batch_size}-w${window_pattern}}"

# Base pretraining
torchrun --standalone --nproc_per_node="${nnpu}" \
  -m scripts.base_train -- \
  --depth="${depth}" \
  --target-param-data-ratio="${data_ratio}" \
  --device-batch-size="${batch_size}" \
  --window-pattern "${window_pattern}" \
  --run="${run_name}"

# Base eval + quick chat
torchrun --standalone --nproc_per_node="${nnpu}" -m scripts.base_eval -- --device-batch-size="${eval_batch_size}"
python -m scripts.chat_cli --source base -p "Why is the sky blue?"

# SFT
torchrun --standalone --nproc_per_node="${nnpu}" -m scripts.chat_sft -- --device-batch-size="${batch_size}" --run="${sft_name}"

# SFT eval + quick chat
torchrun --standalone --nproc_per_node="${nnpu}" -m scripts.chat_eval -- -i sft
python -m scripts.chat_cli --source sft -p "Why is the sky blue?"
