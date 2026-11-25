"""Lightweight smoke test — no torch / model download required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from spect.metrics import (  # noqa: E402
    RunResult,
    format_summary_table,
    save_results,
    summarize,
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "week1_qwen.yaml").read_text(encoding="utf-8"))
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))
    assert cfg["draft_model"].startswith("Qwen/")
    assert cfg["target_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert len(prompts) >= 8
    assert all({"id", "category", "prompt"} <= set(p) for p in prompts)

    fake = [
        RunResult("a", "chat", "baseline", 1.0, 50, 50.0, 10, "x"),
        RunResult("a", "chat", "speculative", 0.5, 50, 100.0, 10, "x"),
        RunResult("b", "code", "baseline", 2.0, 40, 20.0, 10, "y"),
        RunResult("b", "code", "speculative", 1.0, 40, 40.0, 10, "y"),
    ]
    summary = summarize(fake)
    assert abs(summary["speedup"] - 2.0) < 1e-9
    assert summary["meets_week1_target_1_3x"] is True
    assert summary["acceptance_rate"] is None
    assert summary["token_match_rate"] is None

    sweep = [
        RunResult("a", "chat", "baseline", 2.0, 10, 5.0, 4, "base", extra={"new_token_ids": [1, 2]}),
        RunResult(
            "a",
            "chat",
            "speculative",
            1.0,
            10,
            10.0,
            4,
            "same",
            extra={
                "assistant_num_candidate_tokens": 2,
                "new_token_ids": [1, 2],
                "tokens_match": True,
                "draft_tokens": 10,
                "accepted_draft_tokens": 4,
                "verify_steps": 2,
            },
        ),
        RunResult(
            "a",
            "chat",
            "speculative",
            4.0,
            10,
            2.5,
            4,
            "diff",
            extra={
                "assistant_num_candidate_tokens": 4,
                "new_token_ids": [9],
                "tokens_match": False,
                "draft_tokens": 8,
                "accepted_draft_tokens": 2,
                "verify_steps": 2,
            },
        ),
    ]
    sweep_summary = summarize(sweep)
    assert sweep_summary["k_values"] == [2, 4]
    assert sweep_summary["by_k"]["2"]["acceptance_rate"] == 0.4
    assert sweep_summary["by_k"]["2"]["mean_accepted_length"] == 2.0
    assert sweep_summary["by_k"]["2"]["token_match_rate"] == 1.0
    assert sweep_summary["by_k"]["4"]["token_match_rate"] == 0.0
    assert sweep_summary["best_k"] == 2
    assert sweep_summary["by_k"]["2"]["meets_week1_target_1_3x"] is True
    print(format_summary_table(sweep_summary))

    out = ROOT / "results"
    path = save_results(fake, summary, out, "smoke_fake", meta={"smoke": True})
    assert path.is_file()
    print(format_summary_table(summary))
    print(f"[ok] wrote {path}")
    print("[ok] smoke_test_imports passed")


if __name__ == "__main__":
    main()