from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import load_config, run_benchmark


def project_root_from_here() -> Path:
    # src/spect/cli.py -> repo root
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spect Week 1: baseline vs assisted speculative decoding benchmark"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/week1_qwen.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: auto)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override config max_new_tokens",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N prompts (smoke test)",
    )
    p.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup generations",
    )
    p.add_argument(
        "--fallback-target",
        action="store_true",
        help="Force fallback_target_model from config",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root_from_here()
    config_path = Path(args.config)
    if not config_path.is_file():
        config_path = root / args.config
    if not config_path.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    run_benchmark(
        config,
        project_root=root,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        skip_warmup=args.skip_warmup,
        use_fallback_target=args.fallback_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())