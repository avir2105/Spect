# Spect — Speculative Decoding Project Plan

Measurable speculative decoding (same model family, fits ~15GB T4) with baselines, ablations, and a metrics-first demo.

## Goal

Prove wall-clock speedup from speculative / assisted decoding versus standard autoregressive decoding, with honest metrics that would stand up in an interview or portfolio writeup.

## Model family (Week 1 default)

| Role | Model | Why |
|------|--------|-----|
| **Draft** | `Qwen/Qwen2.5-0.5B-Instruct` | Same tokenizer family as target; tiny; fast on T4 |
| **Target** | `Qwen/Qwen2.5-3B-Instruct` (4-bit bitsandbytes) | Comfortable on free T4 with draft; same tokenizer family |

**VRAM budget (approx, T4 15–16GB):**

- Qwen2.5-3B 4-bit ≈ 2–3 GB weights
- Qwen2.5-0.5B fp16 ≈ 1 GB
- KV cache + activations + CUDA overhead → plenty of headroom on 15GB

**Optional / fallbacks:**

1. Larger target: `Qwen/Qwen2.5-7B-Instruct` 4-bit (`configs/week1_qwen_7b.yaml`)
2. Draft: `Qwen/Qwen2.5-0.5B-Instruct` (unchanged)
3. Alternate family: `meta-llama/Llama-3.2-1B-Instruct` draft + `meta-llama/Llama-3.2-3B-Instruct` target (needs HF access)

**Explicitly out of scope for Week 1:** Llama↔Qwen cross-vocab, ngrok remote draft/target, dual-server SD.

---

## Success criteria

### Week 1 (this milestone)

- [x] Project scaffold + runnable benchmark
- [x] Fixed prompt suite (≥8 prompts)
- [x] Baseline greedy vs assisted speculative decoding (HF `assistant_model`)
- [x] Report JSON: tokens/sec, wall time, speedup, by-category breakdown
- [ ] Load draft + target on one GPU (Colab T4) — run `notebooks/week1_colab.ipynb`
- [ ] Target: **≥ 1.3×** tok/s on aggregate (or document why not — still a valid result)

### Week 2 — Metrics product

- Live dashboard / notebook charts: acceptance rate α, mean accepted length τ, draft vs verify time, p50/p95 latency
- Side-by-side baseline vs SD UI

### Week 3 — Research-shaped writeup

- Ablations: speculation length K ∈ {2,4,8}, prompt types (code / chat / short / long)
- Blog-style README section: when SD wins / loses
- Quantization sensitivity notes

### Week 4 — Optional stretch

- Document remote/cross-machine SD failure modes (systems analysis)
- Optional EAGLE head comparison if VRAM allows

---

## Week 1 technical approach

Use Hugging Face **assisted generation** (`assistant_model=` on `generate`):

1. Tokenize fixed prompts
2. Run **baseline**: target only
3. Run **speculative**: target + draft assistant
4. Record wall time, generated tokens, tok/s
5. Write JSON + console summary under `results/`

Lossless under greedy decoding when using HF assisted generation with matching settings — keep decoding config identical across arms.

---

## Repo layout

```
Spect/
  PROJECT_PLAN.md
  README.md
  requirements.txt
  configs/
    week1_qwen.yaml
  data/
    prompts.json
  src/spect/
    __init__.py
    models.py
    decode.py
    metrics.py
    benchmark.py
    cli.py
  notebooks/
    week1_colab.ipynb
  results/          # gitignored outputs
```

---

## How to run Week 1

**Local smoke (CPU / small models — slow):**

```bash
pip install -r requirements.txt
python -m spect.cli --config configs/week1_qwen.yaml --device cpu --max-new-tokens 32 --limit 2
```

**Colab T4 (real numbers):**

1. Open `notebooks/week1_colab.ipynb`
2. Runtime → GPU (T4)
3. Run all cells

**CLI on GPU machine:**

```bash
python -m spect.cli --config configs/week1_qwen.yaml --device cuda --max-new-tokens 128
```

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| OOM (unlikely on 3B) | Shorten `max_new_tokens`; clear cache; avoid loading 7B |
| Colab free quota / session kill | Checkpoint results JSON each prompt; keep notebook self-contained |
| No speedup (<1.3×) | Still ship metrics + analysis; try shorter prompts / higher K / tune candidate tokens |
| HF download failures | Mirror models; cache in Drive on Colab |

---

## Anthropic / portfolio note

This project is strongest as **systems + measurement**: clean baseline, ablations, honest negatives. Anthropic does not offer internships; treat this as portfolio evidence for engineer / Fellows-style applications, not a guarantee of selection.
