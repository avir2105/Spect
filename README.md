# Spect

Speculative / assisted decoding benchmarks with honest metrics.

**Week 1 models (same family, ~15GB T4):**

| Role | Model |
|------|--------|
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` |
| Target | `Qwen/Qwen2.5-3B-Instruct` (4-bit) |

Optional larger target: `configs/week1_qwen_7b.yaml` (`Qwen2.5-7B-Instruct` 4-bit).

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

## Quick start (Colab T4 — recommended)

1. Open [`notebooks/week1_colab.ipynb`](notebooks/week1_colab.ipynb) in Google Colab  
2. Runtime → Change runtime type → **T4 GPU**  
3. Run all cells  

## Local / CLI

```bash
pip install -e ".[gpu]"   # Linux+CUDA; on Windows use pip install -e . and skip bitsandbytes if needed
python -m spect --config configs/week1_qwen.yaml --device cuda
```

Smoke test (2 prompts, short generations):

```bash
python -m spect --config configs/week1_qwen.yaml --limit 2 --max-new-tokens 32
```

Results are written to `results/<config_name>.json` with tok/s, speedup, and per-prompt outputs.

## Week 1 success bar

Aggregate speculative tok/s ÷ baseline tok/s **≥ 1.3×** on the fixed prompt suite — or document why not.

## Layout

```
configs/          YAML run configs
data/prompts.json Fixed evaluation prompts
src/spect/        Loaders, decode, metrics, CLI
notebooks/        Colab Week 1 notebook
results/          Benchmark JSON (gitignored)
```
