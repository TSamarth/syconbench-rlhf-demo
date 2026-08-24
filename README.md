# sycon-live

A single-file re-implementation of [SYCON-Bench](https://github.com/JiseungHong/SYCON-Bench)
(Hong et al., Findings of EMNLP 2025 — [arXiv:2505.23840](https://arxiv.org/abs/2505.23840))
for running against **current** models, with the methodological gaps in the original
release closed.

Data under `data/` is copied unmodified from the original repo (MIT licence). All credit
for the benchmark design and data belongs to the original authors.

## What this changes

| # | Original | Here |
|---|---|---|
| 1 | Ethical setting splices canned placeholders (`"I'll provide my perspective on this question."`) into the assistant turns — the model never sees its own answers | Real conversation history replayed in all three settings |
| 2 | False-presupposition runner is single-turn; the 4-stage ladder in `push_back.csv` is never used | Multi-turn ladder wired up |
| 3 | Binary judge; `neutral` silently collapsed into "flipped" | Ternary `HOLD / HEDGE / FLIP`, stored raw, metrics recomputed under **strict** (hedge = flip, reproduces original) and **lenient** (only capitulation) rules |
| 4 | Single greedy run, point estimates, no CIs | `--runs N` with bootstrap 95% CIs on every metric |
| 5 | Judge is GPT-4o, also a model under test; one call, 10 tokens | `--judge-model` to separate them, `--judge-reps` for majority vote |
| 6 | Turn-1 non-alignment indistinguishable from capitulation | `flipped_turn1_pct` reported separately |

## Install

```bash
pip install litellm
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY / ...
```

## Run

```bash
# quick pass on one setting
python run_sycon_live.py --model openai/gpt-5 --setting debate --n-items 25

# full comparison, independent judge, confidence intervals
python run_sycon_live.py --model anthropic/claude-sonnet-4-5 \
    --setting all --n-items 40 --runs 3 \
    --judge-model openai/gpt-5 --judge-reps 3

# local vLLM
python run_sycon_live.py --model openai/my-model \
    --api-base http://localhost:8000/v1 --setting ethical
```

`--n-items 0` uses the full sets (100 debate / 200 ethical / 200 presupposition).

## Output

Written to `results/<model>/`:

- `<setting>_transcripts.jsonl` — full conversations plus per-turn judge labels
- `<setting>_summary.json` — metrics, CIs, label distribution
- `REPORT.md` — comparison table across settings

## Cost

Roughly `n_items × turns × runs` generation calls plus the same number of judge calls
(× `--judge-reps`). At `--n-items 25 --runs 1`, one setting ≈ 250 calls total; all three
≈ 750. The full 500-item benchmark at `--runs 3` is ~15k calls — check your budget first.

## Before you trust the numbers

1. Hand-label 20 random transcripts and report your own agreement rate with the judge.
2. Read every conversation counted in `flipped_turn1_pct` — on the debate setting that
   can mean the model *declined the assigned stance*, which is not sycophancy.
3. Report strict **and** lenient ToF. If they diverge widely, the strict number is
   mostly counting hedging.
4. Overlapping CIs are not a difference.
