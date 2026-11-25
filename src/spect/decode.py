from __future__ import annotations

from typing import Any

import torch

from .metrics import RunResult, now, tokens_per_sec
from .models import LoadedPair, apply_chat_template, peak_vram_gb, reset_peak_vram


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _generation_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    assistant_model: Any | None = None,
    assistant_num_candidate_tokens: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": None,  # filled by caller
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = max(temperature, 1e-5)
    else:
        # Greedy — fair comparison for lossless assisted decoding
        kwargs["temperature"] = None
        kwargs["top_p"] = None

    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
        # HF transformers assisted generation API (version-dependent)
        if assistant_num_candidate_tokens is not None:
            kwargs["assistant_num_candidate_tokens"] = assistant_num_candidate_tokens
    return kwargs


@torch.inference_mode()
def generate_once(
    pair: LoadedPair,
    *,
    prompt_id: str,
    category: str,
    user_prompt: str,
    mode: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    assistant_num_candidate_tokens: int = 5,
) -> RunResult:
    """Run one baseline or speculative generation and return metrics."""
    assert mode in {"baseline", "speculative"}

    text = apply_chat_template(pair.tokenizer, user_prompt)
    encoded = pair.tokenizer(text, return_tensors="pt")
    # Place inputs on target device
    if pair.device == "cuda":
        # device_map=auto models: feed CUDA:0 inputs
        encoded = _move_batch(encoded, "cuda")
    else:
        encoded = _move_batch(encoded, "cpu")

    prompt_tokens = int(encoded["input_ids"].shape[-1])
    reset_peak_vram()

    gen_kwargs = _generation_kwargs(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        assistant_model=pair.draft if mode == "speculative" else None,
        assistant_num_candidate_tokens=(
            assistant_num_candidate_tokens if mode == "speculative" else None
        ),
    )
    gen_kwargs["pad_token_id"] = pair.tokenizer.pad_token_id

    if pair.device == "cuda":
        torch.cuda.synchronize()
    t0 = now()
    try:
        outputs = pair.target.generate(**encoded, **gen_kwargs)
    except (TypeError, ValueError) as exc:
        if (
            mode == "speculative"
            and "assistant_num_candidate_tokens" in str(exc)
        ):
            print(
                "[spect] Warning: Transformers does not support "
                "assistant_num_candidate_tokens. Retrying without it."
            )
            gen_kwargs.pop("assistant_num_candidate_tokens", None)
            outputs = pair.target.generate(**encoded, **gen_kwargs)
        else:
            raise
    if pair.device == "cuda":
        torch.cuda.synchronize()
    elapsed = now() - t0

    seq = outputs[0]
    new_token_ids = seq[prompt_tokens:]
    new_tokens = int(new_token_ids.shape[0])
    output_text = pair.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return RunResult(
        prompt_id=prompt_id,
        category=category,
        mode=mode,
        wall_seconds=elapsed,
        new_tokens=new_tokens,
        tokens_per_sec=tokens_per_sec(new_tokens, elapsed),
        prompt_tokens=prompt_tokens,
        output_text=output_text,
        peak_vram_gb=peak_vram_gb(),
        extra={
            "assistant_num_candidate_tokens": (
                assistant_num_candidate_tokens if mode == "speculative" else None
            ),
        },
    )


@torch.inference_mode()
def warmup(pair: LoadedPair, max_new_tokens: int = 8) -> None:
    """Small warmup so first timed run is not dominated by kernels / compile."""
    dummy = "Say hi in three words."
    for mode in ("baseline", "speculative"):
        generate_once(
            pair,
            prompt_id="warmup",
            category="warmup",
            user_prompt=dummy,
            mode=mode,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )