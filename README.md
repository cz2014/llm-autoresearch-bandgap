# llm-autoresearch-bandgap

An autonomous LLM research loop (Claude Opus 4.8) that searched for a crystal band-gap
model on `matbench_mp_gap`. The agent edited a single training file, ran it, and kept or
discarded each change against the held-out error, looping until interrupted. Over 184
experiments it kept 7 changes; the resulting model reaches **0.1480 +/- 0.003 eV** mean
absolute error (5-fold), the lowest reported among models trained without external
pretraining (coGN reference: 0.1559 eV).

This repository is a record of that run. It is not set up to re-run end to end; the
preprocessed dataset (`data/`, ~30 GB) is regenerable and not included.

## Files

| File | What it is |
| --- | --- |
| `program.md` | The instructions the agent followed: the goal, the design-space axes A1-A6, the keep-iff-`mae < best - 0.002` rule, and the logging format. |
| `train.py` | The model. The one file the agent edited each experiment. This is its final (champion) state. |
| `results.tsv` | The experiment journal: one row per experiment (commit, MAE, axis, keep/discard, description). 184 rows: 7 kept, 176 discarded, 1 crash. |
| `run.sh` | The launcher: one persistent headless Claude Opus 4.8 session that owned the loop, with a self-heal resume. |
| `prepare.py` | Read-only loader the agent could not edit: the dataset, the official fold, the held-out labels, and the MAE metric. |
| `make_data.py` | One-time script that downloads `matbench_mp_gap` and writes the dataset and the official 5-fold split. |
| `run.log` | Warnings printed during the run. |
| `loop.log` | Launcher log (near-empty). |
| `.gitignore` | Excludes the regenerable dataset and the Python environments. |

### `finalize_5fold/`

Validates the champion to the headline number.

| File | What it is |
| --- | --- |
| `summary_5fold.tsv` | The result: per-fold MAE; mean 0.1480 +/- 0.003 eV. |
| `train_5fold.py` | Trains the champion to convergence on each of the 5 official folds. |
| `train_converge.py` | Trains a single fold to convergence (produces the convergence curve). |
| `converge_curve.csv` | MAE vs training step for the convergence run. |
| `smoke_curve.csv` | Short smoke-test curve. |
| `run_5fold.sh` | Runs the 5 folds. |
| `fold0.log` .. `fold4.log`, `run_5fold*.log`, `converge.log` | Per-run logs. |
| `prepare.py` | Same loader as the parent, kept here so the finalize scripts run standalone. |
