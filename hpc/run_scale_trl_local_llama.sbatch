#!/bin/bash
#SBATCH --job-name=scale-trl-llama
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=scale-trl-llama-%j.out
#SBATCH --error=scale-trl-llama-%j.err
# Override the partition at submit time if needed, e.g.:
#   sbatch --partition=debug hpc/run_scale_trl_local_llama.sbatch
# (#SBATCH directives are not shell-expanded, so this cannot use an env var.)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Adjust for the target cluster's environment manager. On ARM64/Grace-Hopper
# nodes (e.g. Medora) prefer a venv built from the module-provided Python
# rather than assuming Conda is installed:
#   module load python312
#   python3 -m venv ~/llama-venv
#   source ~/llama-venv/bin/activate
#   pip install -r requirements-local-llm.txt
if [ -n "${CONDA_ENV:-}" ]; then
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif [ -n "${VENV_PATH:-}" ]; then
  source "${VENV_PATH}/bin/activate"
fi

: "${MODEL_DIR:?Set MODEL_DIR to the local Llama checkpoint directory}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

srun python co2m/metadata/run_scale_trl_pipeline.py \
  --backend transformers \
  --model "$MODEL_DIR" \
  --device-map auto \
  --torch-dtype "${MODEL_DTYPE:-bfloat16}" \
  --quantization "${MODEL_QUANTIZATION:-none}" \
  --attn-implementation "${MODEL_ATTN_IMPL:-}" \
  --output "${OUTPUT_PATH:-co2m/metadata/scale_trl_two_pass_3899_llama.jsonl}" \
  --report "${REPORT_PATH:-co2m/metadata/scale_trl_two_pass_3899_llama_coverage.json}"
