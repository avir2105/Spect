from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedPair:
    tokenizer: Any
    target: Any
    draft: Any
    device: str
    target_name: str
    draft_name: str
    load_in_4bit: bool


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.float16)


def peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def load_model_pair(
    draft_model: str,
    target_model: str,
    *,
    device: str = "cuda",
    load_in_4bit: bool = True,
    torch_dtype: str = "float16",
    trust_remote_code: bool = True,
) -> LoadedPair:
    """Load same-family draft + target for assisted generation."""
    dtype = _dtype_from_name(torch_dtype)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        print("[spect] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
        use_cuda = False
        load_in_4bit = False

    tokenizer = AutoTokenizer.from_pretrained(
        target_model, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    target_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    draft_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype if use_cuda else torch.float32,
    }

    if use_cuda and load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            target_kwargs["quantization_config"] = quant_config
            target_kwargs["device_map"] = "auto"
        except Exception as exc:
            print(
                f"[spect] 4-bit load unavailable ({exc}); "
                "falling back to fp16 device_map=auto"
            )
            load_in_4bit = False
            target_kwargs["torch_dtype"] = dtype
            target_kwargs["device_map"] = "auto"
    elif use_cuda:
        target_kwargs["torch_dtype"] = dtype
        target_kwargs["device_map"] = "auto"
    else:
        target_kwargs["torch_dtype"] = torch.float32

    print(f"[spect] Loading target: {target_model} (4bit={use_cuda and load_in_4bit})")
    target = AutoModelForCausalLM.from_pretrained(target_model, **target_kwargs)
    target.eval()

    print(f"[spect] Loading draft:  {draft_model}")
    draft = AutoModelForCausalLM.from_pretrained(draft_model, **draft_kwargs)
    if use_cuda and not hasattr(draft, "hf_device_map"):
        draft = draft.to(device)
    elif use_cuda:
        # Prefer placing draft on same device as target inputs
        try:
            draft = draft.to("cuda")
        except Exception:
            pass
    draft.eval()

    # Assisted generation requires matching vocab / special tokens
    if draft.config.vocab_size != target.config.vocab_size:
        raise ValueError(
            f"Vocab mismatch: draft={draft.config.vocab_size} "
            f"target={target.config.vocab_size}. Use same-family models."
        )

    return LoadedPair(
        tokenizer=tokenizer,
        target=target,
        draft=draft,
        device="cuda" if use_cuda else "cpu",
        target_name=target_model,
        draft_name=draft_model,
        load_in_4bit=bool(use_cuda and load_in_4bit),
    )


def apply_chat_template(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_prompt