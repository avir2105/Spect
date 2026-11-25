from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from .decode import generate_once, warmup
from .metrics import RunResult, format_summary_table, save_results, summarize
from .models import load_model_pair


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompts file must be a JSON list")
    return data


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_file():
        return p
    return project_root / p


def candidate_token_values(config: dict[str, Any]) -> list[int]:
    raw = config.get("assistant_num_candidate_tokens", 5)
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw]
    else:
        values = [int(raw)]
    if not values:
        raise ValueError("assistant_num_candidate_tokens must contain at least one K")
    return values


def run_benchmark(
    config: dict[str, Any],
    *,
    project_root: Path,
    device: str | None = None,
    max_new_tokens: int | None = None,
    limit: int | None = None,
    skip_warmup: bool = False,
    use_fallback_target: bool = False,
    candidate_tokens: list[int] | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompts_path = resolve_path(project_root, config["prompts_path"])
    prompts = load_prompts(prompts_path)
    if limit is not None:
        prompts = prompts[:limit]

    target_name = config["target_model"]
    if use_fallback_target and config.get("fallback_target_model"):
        target_name = config["fallback_target_model"]
        print(f"[spect] Using fallback target: {target_name}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_new_tokens = max_new_tokens or int(config.get("max_new_tokens", 128))

    try:
        pair = load_model_pair(
            draft_model=config["draft_model"],
            target_model=target_name,
            device=device,
            load_in_4bit=bool(config.get("load_in_4bit", True)),
            torch_dtype=str(config.get("torch_dtype", "float16")),
        )
    except torch.cuda.OutOfMemoryError:
        if not use_fallback_target and config.get("fallback_target_model"):
            print("[spect] OOM on primary target; retrying with fallback.")
            torch.cuda.empty_cache()
            return run_benchmark(
                config,
                project_root=project_root,
                device=device,
                max_new_tokens=max_new_tokens,
                limit=limit,
                skip_warmup=skip_warmup,
                use_fallback_target=True,
                candidate_tokens=candidate_tokens,
            )
        raise

    ks = candidate_tokens or candidate_token_values(config)
    print(f"[spect] Candidate lengths K={ks}")

    if not skip_warmup:
        print("[spect] Warmup...")
        warmup(
            pair,
            max_new_tokens=min(8, max_new_tokens),
            assistant_num_candidate_tokens=ks[0],
        )

    results: list[RunResult] = []
    total = len(prompts) * (1 + len(ks))
    pbar = tqdm(total=total, desc="benchmark")
    do_sample = bool(config.get("do_sample", False))
    temperature = float(config.get("temperature", 0.0))

    for item in prompts:
        baseline = generate_once(
            pair,
            prompt_id=item["id"],
            category=item.get("category", "unknown"),
            user_prompt=item["prompt"],
            mode="baseline",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            assistant_num_candidate_tokens=ks[0],
        )
        results.append(baseline)
        pbar.set_postfix(
            id=item["id"],
            mode="baseline",
            tps=f"{baseline.tokens_per_sec:.2f}",
            toks=baseline.new_tokens,
        )
        pbar.update(1)
        base_ids = baseline.extra.get("new_token_ids")
        for k in ks:
            result = generate_once(
                pair,
                prompt_id=item["id"],
                category=item.get("category", "unknown"),
                user_prompt=item["prompt"],
                mode="speculative",
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                assistant_num_candidate_tokens=k,
            )
            result.extra["tokens_match"] = result.extra.get("new_token_ids") == base_ids
            results.append(result)
            pbar.set_postfix(
                id=item["id"],
                mode="speculative",
                k=k,
                tps=f"{result.tokens_per_sec:.2f}",
                match="Y" if result.extra["tokens_match"] else "N",
                acc=result.extra.get("acceptance_rate"),
            )
            pbar.update(1)
    pbar.close()

    summary = summarize(results)
    meta = {
        "config_name": config.get("name"),
        "draft_model": pair.draft_name,
        "target_model": pair.target_name,
        "device": pair.device,
        "load_in_4bit": pair.load_in_4bit,
        "max_new_tokens": max_new_tokens,
        "assistant_num_candidate_tokens": ks[0] if len(ks) == 1 else ks,
        "n_prompts": len(prompts),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    raw_out = Path(config.get("output_dir", "results"))
    out_dir = raw_out if raw_out.is_absolute() else project_root / raw_out

    run_name = str(config.get("name", "week1"))
    path = save_results(results, summary, out_dir, run_name, meta=meta)
    print(format_summary_table(summary))
    print(f"\n[spect] Wrote {path}")
    return {"summary": summary, "results": results, "path": str(path), "meta": meta}