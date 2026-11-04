# Local Llama Inference on HPC

The TRL extraction/classification and RAG CLIs can load a Hugging Face model directly into the Python process. This path does not start Ollama, expose an HTTP endpoint, or send document text to a remote service. Ollama remains available with the default `--backend ollama` option.

## Prepare the Model

This implementation targets text Llama 3 Instruct checkpoints such as Llama 3.1 8B Instruct for a single-GPU starting point or Llama 3.3 70B Instruct for a multi-GPU node. Larger checkpoints need substantially more device memory; adjust the Slurm resource request for the cluster. Llama 3 model access may require accepting Meta's license and downloading the checkpoint on a permitted machine before transferring it to the cluster. Llama 4 is not validated by this implementation and may require a different Transformers model class or processor.

Install the cluster-compatible PyTorch build first, then install the optional inference packages with `pip install -r requirements-local-llm.txt`. The checkpoint must already be present on shared storage. Set `MODEL_DIR` to that directory; by default the code uses local files only. The Slurm template assumes a recent GPU with bfloat16 support; set `MODEL_DTYPE=float16` if needed.

## Run the TRL Pipeline

From the repository root, edit resource/environment directives in `hpc/run_scale_trl_local_llama.sbatch` for your cluster, then submit:

```bash
export MODEL_DIR=/shared/models/Meta-Llama-3.1-8B-Instruct
export CONDA_ENV=materials-copilot
sbatch hpc/run_scale_trl_local_llama.sbatch
```

The job writes to a separate `scale_trl_two_pass_3899_llama.jsonl` output. Keep that separate from the existing Qwen output: the resumable runner skips filenames already present in its selected output file.

## Run a Local RAG Query

```bash
python rag_pipeline/rag_query.py \
  --backend transformers \
  --model "$MODEL_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  "What scale was demonstrated for the CO2 conversion process?"
```

The RAG process still needs access to the configured PostgreSQL/pgvector database and embedding model. Only answer generation is switched to local Llama inference.