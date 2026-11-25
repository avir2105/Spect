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

    out = ROOT / "results"
    path = save_results(fake, summary, out, "smoke_fake", meta={"smoke": True})
    assert path.is_file()
    print(format_summary_table(summary))
    print(f"[ok] wrote {path}")
    print("[ok] smoke_test_imports passed")


if __name__ == "__main__":
    main()