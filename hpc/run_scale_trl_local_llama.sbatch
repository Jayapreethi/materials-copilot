#!/bin/bash
#SBATCH --job-name=scale-trl-llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=scale-trl-llama-%j.out
#SBATCH --error=scale-trl-llama-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-materials-copilot}"

: "${MODEL_DIR:?Set MODEL_DIR to the local Llama Instruct checkpoint directory}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

srun python co2m/metadata/run_scale_trl_pipeline.py \
  --backend transformers \
  --model "$MODEL_DIR" \
  --device-map auto \
  --torch-dtype "${MODEL_DTYPE:-bfloat16}" \
  --output co2m/metadata/scale_trl_two_pass_3899_llama.jsonl \
  --report co2m/metadata/scale_trl_two_pass_3899_llama_coverage.json