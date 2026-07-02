# llm-autoresearch-bandgap

A general-purpose coding agent (Claude Code running Claude Opus 4.8) that autonomously optimized
expert-designed crystal graph networks for band-gap prediction. By recombining known methods, the
agent's model reaches a mean absolute error of 0.1480 +/- 0.003 eV on the MatBench benchmark's official
[`matbench_mp_gap`](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_gap/),
which predicts the DFT band gap of a crystal from its structure. This repo is the development history.

## Files
- `program.md` — the agent's instructions: goal, design axes, keep/discard rule, log format.
- `train.py` — the model; the one file the agent edited. Final champion state.
- `results.tsv` — experiment journal, one row each: 184 total (7 kept, 176 discarded, 1 crash).
- `run.sh` — the launcher (one persistent Claude Code session, Opus 4.8).
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
