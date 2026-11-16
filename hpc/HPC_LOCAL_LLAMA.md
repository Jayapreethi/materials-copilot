# Local Llama Inference on HPC

The TRL extraction/classification and RAG CLIs can load a Hugging Face model directly into the Python process. This path does not start Ollama, expose an HTTP endpoint, or send document text to a remote service. Ollama remains available with the default `--backend ollama` option.

## Llama 3 (validated path)

This implementation targets text Llama 3 Instruct checkpoints such as Llama 3.1 8B Instruct for a single-GPU starting point or Llama 3.3 70B Instruct for a multi-GPU node. Larger checkpoints need substantially more device memory; adjust the Slurm resource request for the cluster. Llama 3 model access may require accepting Meta's license and downloading the checkpoint on a permitted machine before transferring it to the cluster.

## Llama 4 (Scout/Maverick)

`local_llm.py` inspects the checkpoint's `config.json` and automatically loads `Llama4ForConditionalGeneration` when the architecture name contains `Llama4`; no code change is needed to switch models. Llama 4 Scout has 109B total parameters (17B active, MoE); its BF16 weights are roughly 218 GB and its FP8 weights roughly 109 GB, neither of which fits a single GPU with 96 GB or less. Pass `--quantization int4` (bitsandbytes 4-bit, ~55 GB for Scout) on `run_scale_trl_pipeline.py` to fit it on one GPU. Verify the quantization backend has a working build for the cluster's CPU architecture (see the Medora notes below) before running the full corpus.

Llama 4 access on Hugging Face is gated separately from Llama 3; request access to the specific Llama 4 model repository even if your account already has Llama 3 approved.


## Medora (UND CRC) specifics

Medora is ARM64 (`aarch64`, Grace Hopper), with Slurm partitions `compute`, `debug`, and `interactive`, each node providing one **NVIDIA GH200 480GB** superchip: ~96 GB GPU-visible HBM3 memory plus 480 GB Grace CPU memory on the same NVLink-C2C coherent domain, and 72 CPU cores. The login node has no GPU; always request a `--gres=gpu:1` allocation (interactively via `srun` or batch via `sbatch`) to run inference.

With only ~96 GB of GPU memory, Llama 4 Scout's BF16 weights (~218 GB) and FP8 weights (~109 GB) do not fit on a single GPU. **Use `--quantization int4`**, which reduces Scout to roughly 55 GB and leaves headroom for activations and KV cache. `bitsandbytes` (int4/int8) is confirmed working on Medora's ARM64 GH200 nodes via `pip install bitsandbytes`.

Medora does not ship a project Conda environment. Build a Python virtual environment instead:

```bash
module load python312
python3 -m venv ~/llama-venv
source ~/llama-venv/bin/activate
pip install -r requirements-local-llm.txt
pip install bitsandbytes  # only if using --quantization int4/int8
```

No PyTorch module is provided on Medora; install it via pip inside the venv (`pip install torch` from the default PyPI index, not the `download.pytorch.org/whl/cuXXX` index, which only hosts x86_64 wheels). Validate GPU visibility with `torch.cuda.is_available()` on an allocated GPU node before downloading large checkpoints.


Install the cluster-compatible PyTorch build first, then install the optional inference packages with `pip install -r requirements-local-llm.txt`. The checkpoint must already be present on shared storage. Set `MODEL_DIR` to that directory; by default the code uses local files only. The Slurm template assumes a recent GPU with bfloat16 support; set `MODEL_DTYPE=float16` if needed.

## Run the TRL Pipeline

From the repository root, edit resource/environment directives in `hpc/run_scale_trl_local_llama.sbatch` for your cluster, then submit:

```bash
export MODEL_DIR=/path/to/Llama-4-Scout-17B-16E-Instruct
export VENV_PATH=$HOME/llama-venv
export MODEL_QUANTIZATION=int4
sbatch --partition=compute hpc/run_scale_trl_local_llama.sbatch
```

For Llama 3 instead, set `MODEL_DIR` to the Llama 3 checkpoint and leave `MODEL_QUANTIZATION` unset.

The job writes to a separate `scale_trl_two_pass_3899_llama.jsonl` output by default (override with `OUTPUT_PATH`/`REPORT_PATH`). Keep that separate from the existing Qwen output: the resumable runner skips filenames already present in its selected output file.

## Run a Local RAG Query

```bash
python rag_pipeline/rag_query.py \
  --backend transformers \
  --model "$MODEL_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  "What scale was demonstrated for the CO2 conversion process?"
```

The RAG process still needs access to the configured PostgreSQL/pgvector database and embedding model. Only answer generation is switched to local Llama inference. The Postgres corpus is not automatically reachable from the cluster; the `CO2M_POSTGRES_DSN` must point to a database the compute node can connect to.
