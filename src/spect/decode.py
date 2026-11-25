from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .metrics import RunResult, now, tokens_per_sec
from .models import LoadedPair, apply_chat_template, peak_vram_gb, reset_peak_vram

_acceptance_hook_warned = False


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

    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
        # Also pass K on generate(). AssistedCandidateGenerator reads it from the
        # draft model's generation_config, which _configure_draft sets.
        if assistant_num_candidate_tokens is not None:
            kwargs["num_assistant_tokens"] = assistant_num_candidate_tokens
            kwargs["num_assistant_tokens_schedule"] = "constant"
    return kwargs


def _configure_decoding(
    pair: LoadedPair,
    *,
    mode: str,
    do_sample: bool,
    temperature: float,
    assistant_num_candidate_tokens: int | None,
) -> None:
    """Make baseline and speculative use the same decoding rule.

    Qwen instruct configs sample by default. Assisted generation draws draft
    tokens from the draft model's own generation_config, and reads the
    candidate length K from that same object.
    """
    target_cfg = pair.target.generation_config
    target_cfg.do_sample = do_sample
    if do_sample:
        target_cfg.temperature = max(temperature, 1e-5)

    if mode != "speculative" or pair.draft is None:
        return
    draft_cfg = pair.draft.generation_config
    draft_cfg.do_sample = do_sample
    if do_sample:
        draft_cfg.temperature = max(temperature, 1e-5)
    if assistant_num_candidate_tokens is not None:
        draft_cfg.num_assistant_tokens = int(assistant_num_candidate_tokens)
        draft_cfg.num_assistant_tokens_schedule = "constant"


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _drafted_from_scores(scores: Any) -> int | None:
    try:
        return max(int(scores.shape[1]) - 1, 0)
    except Exception:
        try:
            return max(int(len(scores[0])) - 1, 0)
        except Exception:
            return None


@contextmanager
def _record_acceptance() -> Iterator[list[tuple[int, int]]]:
    """Record (draft tokens proposed, draft tokens kept) per verification step."""
    steps: list[tuple[int, int]] = []
    try:
        from transformers.generation.candidate_generator import AssistedCandidateGenerator
    except Exception:
        yield steps
        return

    original_get = AssistedCandidateGenerator.get_candidates
    original_update = AssistedCandidateGenerator.update_candidate_strategy
    pending: list[int] = []

    def wrapped_get(self: Any, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        before = int(input_ids.shape[-1])
        out = original_get(self, input_ids, *args, **kwargs)
        pending.append(max(int(out[0].shape[-1]) - before, 0))
        return out

    def wrapped_update(
        self: Any,
        input_ids: torch.Tensor,
        scores: Any,
        num_matches: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        drafted = pending.pop(0) if pending else 0
        if drafted <= 0:
            alt = _drafted_from_scores(scores)
            drafted = alt or 0
        try:
            accepted = _as_int(num_matches)
        except Exception:
            accepted = 0
        if drafted > 0:
            steps.append((drafted, max(0, min(accepted, drafted))))
        return original_update(self, input_ids, scores, num_matches, *args, **kwargs)

    AssistedCandidateGenerator.get_candidates = wrapped_get  # type: ignore[method-assign]
    AssistedCandidateGenerator.update_candidate_strategy = wrapped_update  # type: ignore[method-assign]
    try:
        yield steps
    finally:
        AssistedCandidateGenerator.get_candidates = original_get  # type: ignore[method-assign]
        AssistedCandidateGenerator.update_candidate_strategy = original_update  # type: ignore[method-assign]


def _acceptance_extra(steps: list[tuple[int, int]]) -> dict[str, Any]:
    if not steps:
        return {
            "draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "verify_steps": 0,
            "acceptance_rate": None,
            "mean_accepted_length": None,
            "mean_tokens_per_forward": None,
        }
    drafted = sum(d for d, _ in steps)
    accepted = sum(a for _, a in steps)
    n_steps = len(steps)
    return {
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "verify_steps": n_steps,
        "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
        "mean_accepted_length": round(accepted / n_steps, 4),
        "mean_tokens_per_forward": round((accepted + n_steps) / n_steps, 4),
    }


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

    k = assistant_num_candidate_tokens if mode == "speculative" else None
    _configure_decoding(
        pair,
        mode=mode,
        do_sample=do_sample,
        temperature=temperature,
        assistant_num_candidate_tokens=k,
    )
    gen_kwargs = _generation_kwargs(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        assistant_model=pair.draft if mode == "speculative" else None,
        assistant_num_candidate_tokens=k,
    )
    gen_kwargs["pad_token_id"] = pair.tokenizer.pad_token_id

    if pair.device == "cuda":
        torch.cuda.synchronize()
    t0 = now()
    acceptance_steps: list[tuple[int, int]] = []
    with _record_acceptance() as acceptance_steps:
        try:
            outputs = pair.target.generate(**encoded, **gen_kwargs)
        except (TypeError, ValueError) as exc:
            msg = str(exc)
            if mode == "speculative" and (
                "num_assistant_tokens" in msg
                or "assistant_num_candidate_tokens" in msg
            ):
                print(
                    "[spect] Warning: Transformers rejected the assistant "
                    "candidate-token kwarg. K is still set on the draft "
                    "generation config. Retrying without the kwarg."
                )
                gen_kwargs.pop("num_assistant_tokens", None)
                gen_kwargs.pop("num_assistant_tokens_schedule", None)
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
    extra: dict[str, Any] = {
        "assistant_num_candidate_tokens": k,
        "new_token_ids": [int(t) for t in new_token_ids.tolist()],
    }
    if mode == "speculative":
        extra.update(_acceptance_extra(acceptance_steps))
        if new_tokens > 0 and extra["verify_steps"] == 0:
            global _acceptance_hook_warned
            if not _acceptance_hook_warned:
                print(
                    "[spect] Warning: speculative run produced tokens but "
                    "acceptance was not recorded."
                )
                _acceptance_hook_warned = True

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
        extra=extra,
    )


@torch.inference_mode()
def warmup(
    pair: LoadedPair,
    max_new_tokens: int = 8,
    assistant_num_candidate_tokens: int = 5,
) -> None:
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
            assistant_num_candidate_tokens=assistant_num_candidate_tokens,
        )