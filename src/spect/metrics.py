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


def _k_value(result: RunResult) -> int | None:
    raw = result.extra.get("assistant_num_candidate_tokens")
    if raw is None:
        return None
    return int(raw)


def _acceptance_fields(rows: list[RunResult]) -> dict[str, Any]:
    drafted = accepted = steps = 0
    saw = False
    for row in rows:
        draft_tokens = row.extra.get("draft_tokens")
        accepted_tokens = row.extra.get("accepted_draft_tokens")
        verify_steps = row.extra.get("verify_steps")
        if draft_tokens is None or accepted_tokens is None or verify_steps is None:
            continue
        saw = True
        drafted += int(draft_tokens)
        accepted += int(accepted_tokens)
        steps += int(verify_steps)
    if not saw:
        return {
            "acceptance_rate": None,
            "mean_accepted_length": None,
            "mean_tokens_per_forward": None,
        }
    return {
        "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
        "mean_accepted_length": round(accepted / steps, 4) if steps else None,
        "mean_tokens_per_forward": round((accepted + steps) / steps, 4) if steps else None,
    }


def _token_match_rate(rows: list[RunResult]) -> float | None:
    flags = [row.extra["tokens_match"] for row in rows if "tokens_match" in row.extra]
    if not flags:
        return None
    return round(sum(1 for flag in flags if flag) / len(flags), 4)


def _comparison(results: list[RunResult]) -> dict[str, Any]:
    base = aggregate(results, "baseline")
    spec = aggregate(results, "speculative")
    ratio = speedup(base, spec)
    spec_rows = [row for row in results if row.mode == "speculative"]
    by_category: dict[str, Any] = {}
    for cat in sorted({row.category for row in results}):
        cat_results = [row for row in results if row.category == cat]
        b = aggregate(cat_results, "baseline")
        s = aggregate(cat_results, "speculative")
        cat_spec = [row for row in cat_results if row.mode == "speculative"]
        cat_acc = _acceptance_fields(cat_spec)
        by_category[cat] = {
            "baseline_tps": round(b.tokens_per_sec, 3),
            "speculative_tps": round(s.tokens_per_sec, 3),
            "speedup": round(speedup(b, s), 3),
            "acceptance_rate": cat_acc["acceptance_rate"],
            "mean_accepted_length": cat_acc["mean_accepted_length"],
            "token_match_rate": _token_match_rate(cat_spec),
        }
    return {
        "baseline": asdict(base),
        "speculative": asdict(spec),
        "speedup": round(ratio, 4),
        "meets_week1_target_1_3x": ratio >= 1.3,
        "token_match_rate": _token_match_rate(spec_rows),
        **_acceptance_fields(spec_rows),
        "by_category": by_category,
    }


def summarize(results: list[RunResult]) -> dict[str, Any]:
    spec_rows = [row for row in results if row.mode == "speculative"]
    ks = sorted({k for row in spec_rows if (k := _k_value(row)) is not None})
    if len(ks) <= 1:
        return _comparison(results)

    base_rows = [row for row in results if row.mode == "baseline"]
    by_k: dict[str, Any] = {}
    for k in ks:
        spec_k = [row for row in spec_rows if _k_value(row) == k]
        block = _comparison(base_rows + spec_k)
        block.pop("baseline")
        block["k"] = k
        by_k[str(k)] = block
    best_key = max(by_k, key=lambda key: by_k[key]["speedup"])
    best = by_k[best_key]
    return {
        "baseline": asdict(aggregate(results, "baseline")),
        "k_values": ks,
        "by_k": by_k,
        "best_k": best["k"],
        "speedup": best["speedup"],
        "meets_week1_target_1_3x": any(
            block["meets_week1_target_1_3x"] for block in by_k.values()
        ),
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


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _category_lines(by_category: dict[str, Any]) -> list[str]:
    lines = []
    for cat, stats in by_category.items():
        lines.append(
            f"  {cat:10s}  base={stats['baseline_tps']:.3f}  "
            f"spec={stats['speculative_tps']:.3f}  "
            f"x={stats['speedup']:.3f}  "
            f"accept={_fmt_rate(stats.get('acceptance_rate'))}  "
            f"match={_fmt_rate(stats.get('token_match_rate'))}"
        )
    return lines


def format_summary_table(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    lines = [
        "",
        "=" * 72,
        "BENCHMARK SUMMARY",
        "=" * 72,
        f"Baseline    : {b['tokens_per_sec']:.3f} tok/s  "
        f"({b['total_new_tokens']} tokens / {b['total_wall_seconds']:.2f}s)",
    ]
    if "by_k" in summary:
        for key, block in summary["by_k"].items():
            s = block["speculative"]
            lines.extend(
                [
                    "-" * 72,
                    f"K={key}  spec={s['tokens_per_sec']:.3f} tok/s  "
                    f"({s['total_new_tokens']} tokens / {s['total_wall_seconds']:.2f}s)",
                    f"     speedup={block['speedup']:.3f}x  "
                    f"≥1.3x={'YES' if block['meets_week1_target_1_3x'] else 'NO'}  "
                    f"accept={_fmt_rate(block.get('acceptance_rate'))}  "
                    f"mean_len={_fmt_rate(block.get('mean_accepted_length'))}  "
                    f"match={_fmt_rate(block.get('token_match_rate'))}",
                    *_category_lines(block.get("by_category", {})),
                ]
            )
        lines.append("-" * 72)
        lines.append(
            f"Best K={summary['best_k']}  speedup={summary['speedup']:.3f}x  "
            f"any K ≥1.3x={'YES' if summary['meets_week1_target_1_3x'] else 'NO'}"
        )
    else:
        s = summary["speculative"]
        lines.extend(
            [
                f"Speculative : {s['tokens_per_sec']:.3f} tok/s  "
                f"({s['total_new_tokens']} tokens / {s['total_wall_seconds']:.2f}s)",
                f"Speedup     : {summary['speedup']:.3f}x",
                f"≥1.3x goal  : {'YES' if summary['meets_week1_target_1_3x'] else 'NO'}",
                f"Acceptance  : {_fmt_rate(summary.get('acceptance_rate'))}  "
                f"mean accepted length={_fmt_rate(summary.get('mean_accepted_length'))}",
                f"Token match : {_fmt_rate(summary.get('token_match_rate'))}",
                "-" * 72,
                "By category:",
                *_category_lines(summary.get("by_category", {})),
            ]
        )
    lines.append("=" * 72)
    return "\n".join(lines)