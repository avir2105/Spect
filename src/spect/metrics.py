from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    prompt_id: str
    category: str
    mode: str  # baseline | speculative
    wall_seconds: float
    new_tokens: int
    tokens_per_sec: float
    prompt_tokens: int
    output_text: str
    peak_vram_gb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateStats:
    mode: str
    n: int
    total_new_tokens: int
    total_wall_seconds: float
    tokens_per_sec: float
    mean_tokens_per_sec: float
    median_tokens_per_sec: float
    mean_wall_seconds: float


def now() -> float:
    return time.perf_counter()


def tokens_per_sec(new_tokens: int, wall_seconds: float) -> float:
    if wall_seconds <= 0:
        return 0.0
    return new_tokens / wall_seconds


def aggregate(results: list[RunResult], mode: str) -> AggregateStats:
    subset = [r for r in results if r.mode == mode]
    if not subset:
        return AggregateStats(
            mode=mode,
            n=0,
            total_new_tokens=0,
            total_wall_seconds=0.0,
            tokens_per_sec=0.0,
            mean_tokens_per_sec=0.0,
            median_tokens_per_sec=0.0,
            mean_wall_seconds=0.0,
        )
    total_tokens = sum(r.new_tokens for r in subset)
    total_wall = sum(r.wall_seconds for r in subset)
    tps_list = [r.tokens_per_sec for r in subset]
    return AggregateStats(
        mode=mode,
        n=len(subset),
        total_new_tokens=total_tokens,
        total_wall_seconds=total_wall,
        tokens_per_sec=tokens_per_sec(total_tokens, total_wall),
        mean_tokens_per_sec=statistics.mean(tps_list),
        median_tokens_per_sec=statistics.median(tps_list),
        mean_wall_seconds=statistics.mean(r.wall_seconds for r in subset),
    )


def speedup(baseline: AggregateStats, speculative: AggregateStats) -> float:
    if baseline.tokens_per_sec <= 0:
        return 0.0
    return speculative.tokens_per_sec / baseline.tokens_per_sec


def summarize(results: list[RunResult]) -> dict[str, Any]:
    base = aggregate(results, "baseline")
    spec = aggregate(results, "speculative")
    ratio = speedup(base, spec)
    by_category: dict[str, Any] = {}
    categories = sorted({r.category for r in results})
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        b = aggregate(cat_results, "baseline")
        s = aggregate(cat_results, "speculative")
        by_category[cat] = {
            "baseline_tps": round(b.tokens_per_sec, 3),
            "speculative_tps": round(s.tokens_per_sec, 3),
            "speedup": round(speedup(b, s), 3),
        }
    return {
        "baseline": asdict(base),
        "speculative": asdict(spec),
        "speedup": round(ratio, 4),
        "meets_week1_target_1_3x": ratio >= 1.3,
        "by_category": by_category,
    }


def save_results(
    results: list[RunResult],
    summary: dict[str, Any],
    output_dir: str | Path,
    run_name: str,
    meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_name}.json"
    payload = {
        "meta": meta or {},
        "summary": summary,
        "runs": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_summary_table(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    s = summary["speculative"]
    lines = [
        "",
        "=" * 60,
        "WEEK 1 BENCHMARK SUMMARY",
        "=" * 60,
        f"Baseline    : {b['tokens_per_sec']:.3f} tok/s  "
        f"({b['total_new_tokens']} tokens / {b['total_wall_seconds']:.2f}s)",
        f"Speculative : {s['tokens_per_sec']:.3f} tok/s  "
        f"({s['total_new_tokens']} tokens / {s['total_wall_seconds']:.2f}s)",
        f"Speedup     : {summary['speedup']:.3f}x",
        f"≥1.3x goal  : {'YES' if summary['meets_week1_target_1_3x'] else 'NO'}",
        "-" * 60,
        "By category:",
    ]
    for cat, stats in summary.get("by_category", {}).items():
        lines.append(
            f"  {cat:10s}  base={stats['baseline_tps']:.3f}  "
            f"spec={stats['speculative_tps']:.3f}  "
            f"x={stats['speedup']:.3f}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)