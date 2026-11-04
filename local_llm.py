"""In-process Hugging Face Transformers inference for local/HPC model weights."""

from __future__ import annotations

import json
from typing import Any


class LocalTransformersModel:
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Local inference requires PyTorch and Transformers. Install them in the HPC environment."
            ) from error

        dtype = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

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
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_options)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        schema: dict[str, Any] | None = None,
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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()