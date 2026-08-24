# SYCON-Bench demo — setup & run recipe

*This is the API-only path. We run `run_sycon_live.py` — a single-file re-implementation of
SYCON-Bench that closes the original release's methodological gaps (see `CODE-AUDIT.md`).
We do **not** run the original JiseungHong/SYCON-Bench repo on camera: it hardcodes private
CMU/Azure endpoints, its false-presupposition runner is single-turn only, and its ethical
runner never shows the model its own answers (audit §B). The re-implementation was verified
end-to-end **offline** here on 2026-08-22 — data loading and metrics confirmed; the live API
generation/judge path is untested until you add keys (expect one or two provider-specific
edges on first run).*

## 0. What you already have

This folder is self-contained. No clone needed.

- `run_sycon_live.py` — the harness.
- `data/` — the benchmark data, copied unmodified from the original repo (MIT). Verified
  present and schema-correct: **100** debate items, **200** ethical, **200** false-
  presupposition; all fp questions match a pushback row (0 silently skipped).

## 1. Environment (Python 3.10+)

```powershell
# Windows / PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt      # this is just: litellm
```

`litellm` is the only dependency. Everything else the harness uses is standard library.

Set the provider key(s) for the model(s) you'll run, before running:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), etc. Only
the providers you actually use are needed. Any litellm-routable model string works.

## 2. Smoke-test one cheap call first

Before any real run, confirm your key and model string route correctly:

```bash
python run_sycon_live.py --model openai/gpt-5-mini --setting debate --n-items 2 --judge-model openai/gpt-5-mini
```

Two items, one setting. If that writes a `REPORT.md` under `results/`, wiring is good.

## 3. The run recipe

```bash
# quick, defensible pass on the two on-thesis settings, independent judge, CIs
python run_sycon_live.py --model openai/gpt-5 \
    --setting debate --n-items 40 --runs 3 \
    --judge-model anthropic/claude-sonnet-4-5 --judge-reps 3

python run_sycon_live.py --model openai/gpt-5 \
    --setting presupposition --n-items 40 --runs 3 \
    --judge-model anthropic/claude-sonnet-4-5 --judge-reps 3
```

Key flags:

- `--setting` — `debate`, `ethical`, `presupposition`, or `all`. **Prefer `debate` (the
  cold-open clip) and `presupposition` (the clean "caved on a false fact" number). Leave
  `ethical` off-camera** (see `PURPOSE.md`).
- `--n-items 0` uses the full set (100 / 200 / 200). Default is 25.
- `--runs N` (N>1) samples repeatedly at `--temperature` and puts **95% bootstrap CIs** on
  every metric. `--runs 1` is a single greedy (temp 0) pass with no CI. Use ≥3 for any
  number that goes on screen.
- `--judge-model` — **always set this to a different model than the one under test.** The
  harness warns if judge == model (self-preference bias). Use a strong, cheap third model.
- `--judge-reps 3` — majority-vote judging, blunts single-call judge variance.
- `--workers` — parallelism (default 4). Raise cautiously; watch provider rate limits.

## 4. Output

Written to `results/<model>/`:

- `<setting>_transcripts.jsonl` — every conversation, with per-turn HOLD/HEDGE/FLIP labels.
- `<setting>_summary.json` — metrics under **both** strict and lenient rules, CIs, and the
  raw label distribution.
- `REPORT.md` — the comparison table across whatever settings you ran.

**The strict/lenient split is the point.** `strict` counts HEDGE as a flip (reproduces the
original metric); `lenient` counts only outright capitulation. A wide gap means the strict
number is mostly counting hedging, not caving. Report both.

## 5. Cost

Roughly `n_items × turns × runs` generation calls, plus the same number of judge calls
(× `--judge-reps`). At `--n-items 25 --runs 1`, one setting ≈ 250 calls; all three ≈ 750.
The full 500-item benchmark at `--runs 3` is ~15k generation + ~45k judge calls — **estimate
spend before running that.** For a demo, `--n-items 40 --runs 3` on two settings is plenty.

## 6. Before any number goes on camera

1. **Hand-label 20 random transcripts** and report your own agreement rate with the judge.
   The judge is an LLM; this is the honesty check the original benchmark never shipped.
2. **Read every conversation counted in `flipped_turn1_pct`.** On debate this can mean the
   model *declined the assigned stance* at turn 1 — good behaviour, not sycophancy — scored
   the same as instant capitulation (audit B.2).
3. **Report strict and lenient ToF.** If they diverge widely, the strict number is mostly
   hedging.
4. **Overlapping CIs are not a difference.** Do not rank two models whose intervals overlap.
5. **A failed generation is scored FLIP.** If the model API errors out through all retries,
   the turn is empty and the judge labels it FLIP — which inflates the flip rate. Grep
   transcripts for empty `responses` before quoting a number (see `CODE-AUDIT.md`, residual
   caveat R2).
6. **Runs are non-deterministic** at `--runs > 1` (temperature sampling). Re-running won't
   reproduce byte-identical percentages — say so if asked.
