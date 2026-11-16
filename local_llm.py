"""In-process Hugging Face Transformers inference for local/HPC model weights."""

from __future__ import annotations

import json
from typing import Any

QUANTIZATION_CHOICES = ("none", "int4", "int8", "fp8")


def _resolve_model_class(model_path: str, local_files_only: bool):
    """Pick the model class based on the checkpoint's own config, not a user guess."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    architectures = getattr(config, "architectures", None) or []
    if any("Llama4" in name for name in architectures):
        from transformers import Llama4ForConditionalGeneration

        return Llama4ForConditionalGeneration, True
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM, False


def _build_quantization_config(quantization: str):
    if quantization == "none":
        return None
    if quantization in ("int4", "int8"):
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=quantization == "int4",
            load_in_8bit=quantization == "int8",
            bnb_4bit_compute_dtype="bfloat16",
        )
    if quantization == "fp8":
        from transformers import FbgemmFp8Config

        return FbgemmFp8Config()
    raise ValueError(f"Unsupported quantization: {quantization!r}; choose from {QUANTIZATION_CHOICES}")


class LocalTransformersModel:
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        local_files_only: bool = True,
        quantization: str = "none",
        attn_implementation: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Local inference requires PyTorch and Transformers. Install them in the HPC environment."
            ) from error

        dtype = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

        model_class, self.is_llama4 = _resolve_model_class(model_path, local_files_only)

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        model_options = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
        }
        if device_map.lower() == "cpu":
            model_options["low_cpu_mem_usage"] = False
        else:
            model_options["device_map"] = device_map
            model_options["low_cpu_mem_usage"] = True
        quantization_config = _build_quantization_config(quantization)
        if quantization_config is not None:
            model_options["quantization_config"] = quantization_config
        if attn_implementation:
            model_options["attn_implementation"] = attn_implementation
        self.model = model_class.from_pretrained(model_path, **model_options)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        schema: dict[str, Any] | None = None,
        min_new_tokens: int = 16,
    ) -> str:
        if schema is not None:
            system_prompt += (
                "\n\nReturn exactly one valid JSON object matching this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = model_inputs["input_ids"]
        input_device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(input_device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min(min_new_tokens, max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()