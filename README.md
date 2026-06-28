# llm-autoresearch-bandgap

An autonomous LLM research loop (Claude Opus 4.8) that searched `matbench_mp_gap` over 184
experiments, kept 7 changes, and reached 0.1480 +/- 0.003 eV MAE (5-fold). A record of the
run; the regenerable dataset (~30 GB) is not included.

## Files
- `program.md` — the agent's instructions: goal, design axes, keep/discard rule, log format.
- `train.py` — the model; the one file the agent edited. Final champion state.
- `results.tsv` — experiment journal, one row each: 184 total (7 kept, 176 discarded, 1 crash).
- `run.sh` — the launcher (one persistent Opus session).
- `prepare.py` — read-only data / fold / metric loader.
- `make_data.py` — downloads `matbench_mp_gap`, writes the dataset and official 5-fold split.
- `run.log`, `loop.log` — run logs.
- `.gitignore` — excludes the dataset and environments.

## `finalize_5fold/`
5-fold validation of the champion.
- `summary_5fold.tsv` — per-fold MAE; mean 0.1480 +/- 0.003 eV.
- `train_5fold.py` — trains the champion on each official fold.
- `train_converge.py`, `converge_curve.csv` — single-fold convergence run and its curve.
- `run_5fold.sh`, `smoke_curve.csv`, `*.log` — driver, smoke curve, per-run logs.
- `prepare.py` — loader copy so the scripts run standalone.
