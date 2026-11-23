# Axiom

Autonomous root-cause analysis for e-commerce business metrics.

Dashboards tell you *that* revenue dropped. Axiom works out *why*. For each day of event data it
checks data quality, detects which KPIs moved significantly, decomposes the metric tree, drills
through dimensions (platform, country, payment method, app version, ...) and produces
evidence-backed hypotheses about the cause.

> "Something changed. Figure out what happened, why it happened, and tell me whether I need to act."

**Status:** M0–M8 built. Runs fully offline on a local model (Qwen2.5 7B via Ollama, no API key) and has been
run on a real dataset (UCI Online Retail II). A fine-tuning pipeline for a smaller custom model is included. See [Team](#team) for who owns what, [HANDOFF.md](HANDOFF.md) for detailed status and [PRD.md](PRD.md) for the full spec.

## How it works

```
Ingestion → Data Quality (PASS/WARN/BLOCK) → Detection → Investigation (LLM + tools)
   → Critic (≤2 revise loops) → Reporting
```

The design rule is that the deterministic core computes the numbers and the LLM only reasons and
writes:

1. The LLM never computes numbers. Python/DuckDB does.
2. Every claim cites an `evidence_id`. A claim with no citation, or with a number that can't be
   traced to its cited evidence, is dropped ([axiom/evidence/verifier.py](axiom/evidence/verifier.py)).
3. If data quality is `BLOCK`, there is no analysis, only a data-quality alert.
4. The Critic can veto a hypothesis; a `revise` verdict sends the hypothesis back to investigation (max 2 loops).
5. Runs are reproducible: seeded data and cached LLM responses (temperature 0).
6. Contributions add up to at most 100%, and the unexplained remainder is shown.

Axiom ships with its own benchmark: a simulator that generates synthetic sessions, events and
orders with **28 injected scenarios** (payment failures, traffic drops, app-version bugs, stock-outs,
Simpson's-paradox mix shifts, two-cause days, tracking bugs, schema changes, late data, and
no-anomaly controls) plus a `ground_truth.json` to score against.

## Quick start

Requires Python 3.11+.

```bash
pip install -e .[dev]

# Generate 180 days of synthetic data (~15 s, ~90 MB) into data/sim/
axiom simulate --out data/sim

# Lint and test (the data-dependent tests skip if data/sim is missing)
ruff check .
pytest
```

The LLM steps run on a local model by default. Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:7b      # ~4.7 GB, runs on a 6 GB GPU
axiom run 2026-04-01        # S11, electronics stock-out: ~4 min on an RTX 4050
axiom ui                    # open http://localhost:8501
```

`simulate` options: `--days`, `--sessions` (per day, default 10k), `--seed`, `--no-plot`. It
writes one parquet file per table per day, `ground_truth.json` and a `revenue.png` with the
scenario days marked.

### Commands

```bash
axiom run 2026-04-01                       # one day: DQ -> detect -> investigate -> critic -> report
axiom eval --systems stats_only --seeds 42,43,44          # offline benchmark, no LLM
axiom eval --scenarios S07,S11 --systems axiom,no_critic,baseline_a   # needs an LLM provider
axiom ui                                   # Streamlit: runs, reports, evidence behind each claim, eval dashboard
axiom daily --data data/sim                # scheduled run for yesterday; Slack ping if AXIOM_SLACK_WEBHOOK is set
axiom adapt-online-retail                  # real UCI Online Retail II data -> data/online_retail
axiom adapt-olist path/to/olist_csvs       # Olist (Kaggle) CSVs -> data/olist
axiom train-data                           # random simulated incidents -> expert trajectories (data/train)
axiom train                                # LoRA fine-tune Qwen2.5-1.5B on them (CUDA GPU, ~2-3 h)
```

Schedule `axiom daily` with cron (`0 7 * * *`) or Windows Task Scheduler.

### LLM provider

`config/settings.yaml` picks the model and backend:

| Backend | Settings | Notes |
|---|---|---|
| **Ollama (default)** | `model: qwen2.5:7b`, `provider: {base_url: "http://localhost:11434/v1"}` | Local, free, no key |
| Any OpenAI-compatible API | `provider: {base_url, api_key_env}` | Key read from the named env var, never stored |
| Fine-tuned local model | `provider: {type: local, adapter: models/axiom-lora, base: Qwen/Qwen2.5-1.5B-Instruct}` | After `axiom train` |
| Anthropic | remove `provider`, set a `claude-*` model | Needs `ANTHROPIC_API_KEY` |

All backends go through the same tool-use loop ([axiom/llm/](axiom/llm/)) and the same response cache.

### Training a custom model

`axiom train-data` simulates 30 datasets with a random incident every 5 days (payment failures, traffic drops,
funnel bugs, price changes, stock-outs, mix shifts, two-cause days). A teacher that knows the injected cause plays
the investigation (the true dimensions plus decoys, so the model learns to read tool outputs) and the Critic
verdict, writing the exact conversations the model will see. `axiom train` LoRA-fine-tunes Qwen2.5-1.5B on them;
the loss is taken only on the model's own turns. The 28 hand-written scenarios are never used for training, so
they stay a fair held-out test.

### Using the pieces from Python

```python
from datetime import date
from pathlib import Path

from axiom.quality import check_day
from axiom.detection import detect
from axiom.analysis.tools import Tools

data, day = Path("data/sim"), date(2026, 4, 1)   # S11: electronics stock-out
print(check_day(data, day).status)                # PASS / WARN / BLOCK

tools = Tools(data)
anomalies = detect(data, day, tools.store)        # KPIs that moved, with evidence ids
print(tools.find_causes(str(day)))                # deterministic tree walk + drill-down (Baseline C)
```

To run the LLM investigation agent directly, call
`axiom.agents.investigation.investigate(tools, LLMClient(), day, anomalies)`; the backend comes from
[config/settings.yaml](config/settings.yaml).

## Project layout

| Path | What it is |
|---|---|
| `axiom/simulator/` | Synthetic data generator and scenario injector |
| `axiom/ingestion/` | Per-day parquet files → DuckDB views; Olist and Online Retail II adapters |
| `axiom/quality/` | Data-quality checks: schema diff, volume, nulls, duplicates, category drift, tracking invariants |
| `axiom/detection/` | Robust-z change detection against weekday and trailing-7-day baselines |
| `axiom/analysis/` | Metric cube, metric-tree decomposition (LMDI), Simpson-safe drill-down, LLM-facing tools |
| `axiom/evidence/` | Evidence store and numeric-hallucination verifier |
| `axiom/llm/` | LLM client (response cache, token tally) with Anthropic, OpenAI-compatible/Ollama and fine-tuned local backends |
| `axiom/agents/` | Investigation agent, Critic, report writer and their prompts |
| `axiom/baselines/`, `axiom/eval/` | Baseline A, No-Critic and stats-only systems; benchmark harness |
| `axiom/train/` | Random incidents, teacher trajectories and LoRA fine-tuning |
| `axiom/ui/` | Streamlit app |
| `config/` | Settings, registered schema, metric tree |
| `scenarios/` | Scenario definitions (YAML) |
| `tests/` | Pytest suite |

## Results

Benchmark of the deterministic baseline (**Baseline C / `stats_only`**: DQ gate + detection + tree walk + drill-down,
no LLM) on all 28 scenarios x 3 seeds (84 runs, `eval/results/stats_only_3seeds/`). Match rule: the predicted filter
set must contain every true (dimension, value) pair ([D27](DECISIONS.md)).

| Metric | stats_only |
|---|---|
| Detection precision / recall | 0.85 / 0.83 |
| False-positive rate (no-anomaly days) | 0.08 |
| Root-cause accuracy@1 / @3 (strict) | 0.50 / 0.50 |
| Lenient overlap@3 (any true pair found) | 0.75 |
| Both causes found (two-cause scenarios) | 0.00 |
| Contribution error | 37.9 pp |
| Data-issue recall / false flags | 1.00 / 0.00 |
| Latency (median) | 0.9 s |

**Axiom, No-Critic and Baseline A rows are still empty.** The harness and all three systems are built and
unit-tested with a scripted LLM, but the only LLM available during development was a rate-limited free tier,
which could not complete enough runs to report. Run
`axiom eval --systems axiom,no_critic,baseline_a,stats_only --seeds 42,43,44` with a provider that has
capacity and paste the table here. Until then there is no evidence that Axiom beats Baseline A.

### Real data: UCI Online Retail II

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) is the actual transaction log of a UK
online retailer, Dec 2009 to Dec 2011 (CC BY 4.0, no login). `axiom adapt-online-retail` turns it into 604 trading
days; detection flags 81 of them. Two runs with the local Qwen 7B model:

| Day | What moved | Axiom's answer | Check against the data |
|---|---|---|---|
| 2011-03-17 | revenue +57% (z=15) | bags category, 61% of the change | correct: bags £2.0k → £11.0k |
| 2010-04-19 | revenue +71% (z=16) | broad traffic increase | partly: guest checkouts went £2.6k → £20.7k (96%), which the 7B model missed |

## Limitations

- **No full benchmark of the LLM systems yet** (above). Live evidence so far: S11 solved by DeepSeek-V4-Flash
  (API) and by Qwen2.5 7B (local); a 2-scenario run is in `eval/results/live_llm_small/`.
- **The Critic's verdict often doesn't arrive on Ollama**, which does not enforce a forced tool call; the hypothesis
  is then kept but flagged `UNREVIEWED` with confidence capped at 0.3. The fine-tuned model is trained to answer it.
- **The fine-tuned model is not trained yet** in this repo (no weights are committed); run `axiom train-data` and
  `axiom train` on a CUDA GPU.
- **Synthetic data.** All numbers above are on the simulator, whose scenarios the detector was tuned against.
  Scenarios with a true effect under ~6% (S02, S07, S08, S12) sit below the revenue noise floor ([D20](DECISIONS.md)).
- **Stats-only misses two-dimension causes** (S01, S05) and second causes (S23); the strict metric reflects that.
- **Contribution error is approximate** ([D30](DECISIONS.md)); per-segment ground truth is not stored.
- **Real datasets have no sessions, funnel or payment data** (Olist, Online Retail II), so traffic and conversion
  metrics are degenerate there and only revenue, orders, AOV and their dimensions are meaningful. DQ thresholds are
  tuned for the simulator.
- **Baseline A sandbox** has no network or memory limit on Windows ([D29](DECISIONS.md)).
- `config/` is read relative to the source tree, so Axiom only works from a checkout or editable install.

## Team

### Done so far

- **M0** Skeleton: repo layout, Pydantic models, CLI, config, CI
- **M1** Simulator: data generator, scenario injector, 28 scenarios, ground truth
- **M2** Ingestion and data-quality checks
- **M3** Detection, metric-tree decomposition, drill-down, analysis tools, evidence store
- **M4** LLM client and investigation agent with the numeric verifier (tested against a scripted fake LLM)
- **M5** Critic agent (6 checks), report writer (Markdown + JSON), orchestrator with revise loop, `axiom run` wired
- **M6** Baseline A (sandboxed pandas agent), No-Critic ablation, stats-only baseline, `axiom eval` harness
- **M7** Streamlit UI, `axiom daily` scheduler entry point, Slack webhook
- **M8** Olist and Online Retail II adapters, real-data runs, results and limitations write-up
- **Local model** Ollama backend (default), fine-tuning pipeline (`axiom train-data`, `axiom train`)

### Remaining work

| Item | Scope |
|---|---|
| Fine-tuned model | Run `axiom train-data` + `axiom train`, then benchmark it against Qwen 7B and stats-only |
| LLM benchmark | Run `axiom eval` for the LLM systems on all 28 scenarios and fill the results table |
| Real data | Review DQ thresholds on real data; try Olist (needs a Kaggle login) |

Design decisions and their reasoning are logged in [DECISIONS.md](DECISIONS.md).
