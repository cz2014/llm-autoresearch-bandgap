# autoresearch -- mp_gap (representation-first)

Goal: the LOWEST `mae:` (eV) on matbench_mp_gap, on the held-out official fold, under a fixed compute
budget. You edit ONE file (`train.py`), run it, and keep or discard the change. Loop until interrupted.

## Setup (once)
1. Branch `autoresearch/claude-jun16` (`git checkout -b autoresearch/claude-jun16` from master if needed). Never
   commit to master.
2. Read `prepare.py` (READ-ONLY: data, official fold, private TEST labels, MAE metric -- never edit it,
   never read `data/*` directly) and `train.py` (the ONLY file you edit).
3. Skim the reference architectures in `ref/` before your first idea -- real, installed code on this box:
   - `ref/matgl/src/matgl/graph/_compute.py` -- IMPORTABLE angle/line-graph utilities you already have:
     `compute_theta_and_phi`, `create_line_graph`, `_compute_3body_indices`. (train.py already imports
     `compute_pair_vector_and_distance` from here.)
   - `ref/matgl/src/matgl/models/_m3gnet.py` (three-body / bond angles), `_megnet.py` (global-state block).
   - `ref/matbench/benchmarks/matbench_v0.1_{coGN,alignn}/` -- rank-1 coGN (0.1559) + ALIGNN:
     connectivity + hyperparameters.
   Take ideas; do not copy (the matched-compute envelope applies).
4. Create `results.tsv` with the header if missing (see Logging).

## Design space -- spread experiments ACROSS these axes
  A1 CONNECTIVITY -- radius cutoff (now 5.0), k-NN, Voronoi, multi-graph.
  A2 GEOMETRY / ANGLES (highest lever) -- distance-only now; add BOND ANGLES via a line graph
     (ALIGNN / M3GNet three-body), dihedrals, Bessel/Gaussian radial basis. Distance-only radius graphs
     cap accuracy on mp_gap.
  A3 FEATURES -- node (element table, oxidation, electronegativity), edge (RBF), GLOBAL state
     (composition, density, symmetry) fed once or updated per layer (MEGNet-style).
  A4 ARCHITECTURE -- message-passing block (CGConv now -> MEGNet / gated / attention / three-body),
     depth, width, norm.
  A5 READOUT -- mean+max pool now -> set2set, attention, weighted / sum.
  A6 OPTIMIZATION -- loss, EMA, weight decay, dropout, schedule, batch size. Mostly mapped out
     (floor ~0.1755); prefer A1-A5.

If several experiments in a row stall, a good move is to switch to an axis you have not tried lately
(A2 angles or A1 connectivity are the strongest levers) and take a concrete idea from a `ref/` file,
rather than tuning another A6 knob.

## CAN / CANNOT
CAN (all in `train.py`): everything in A1-A6 -- featurization (cutoff, connectivity, line graphs and
bond angles, node/edge/global features, multiple graphs per crystal), model, optimizer, schedule,
training loop, batch size, readout.
CANNOT: edit `prepare.py`; change the data / fold split / TEST labels / `prepare.evaluate_mae` metric;
read `data/*` directly (data ONLY via `prepare.train_set` / `prepare.test_structures` /
`prepare.evaluate_mae`); install packages (read/import from what is already installed); exceed
5,000,000 params (print `num_params`); touch `run.sh`, tmux, or any process but `python train.py`; end
this session.

## Hard constraints
1. ENVELOPE -- matched, not scaled: <= 5M params, single GPU, FP16 ok. A win must be
   algorithmic/representational. Heavier graphs (line / three-body) train fewer steps in the fixed
   budget -- that is the rule; if an angle model is close but under-trained, shrink it rather than drop
   the axis.
2. FIXED BUDGET -- each run trains `prepare.TIME_BUDGET` s (1500) then evaluates. Keep the cosine horizon
   coupled to step count (`SCHED_TMAX` from the budget; a fixed `T_max` ramps the LR back up).
3. FIXED SEED -- `train.py` seeds RNG at the top so one run is a fair comparison. Keep it.
4. FEAT CACHE -- `featurize()` caches per `FEAT_KEY` under `prepare.cache_dir(FEAT_KEY)`. CHANGE
   `FEAT_KEY` whenever featurize() output changes, or you read a stale cache. `feat_seconds` is printed
   and does NOT count against the budget.

## Run
```
source .venv/bin/activate
python train.py > run.log 2>&1
grep "^mae:\|^num_params:\|^feat_seconds:\|^peak_vram_mb:" run.log   # empty mae => crash (tail run.log)
```

## Loop (forever)
1. Check git state + recent results.tsv axes.
2. One hypothesis; pick its axis deliberately; edit `train.py`.
3. `git commit -am "<axis>: <hypothesis>"`.
4. Run; read `grep "^mae:" run.log`.
5. KEEP iff `mae < best - 0.002` (advance; `best := mae`); else DISCARD: `git reset --hard` to best.
   (The 0.002 margin skips sub-noise changes; the seed is fixed, so one run decides.)
6. At equal mae, simpler wins; deleting code for equal/better mae is a great result.
7. Never stop, never ask whether to continue. Out of ideas in an axis -> switch axes (consult `ref/`).

## Logging (results.tsv, TAB-separated, NOT committed)
```
commit	mae	memory_gb	axis	status	description
```
commit (7-hash) | mae (0.000000 if crash) | peak_vram_mb/1024 .1f | axis A1..A6 | keep|discard|crash |
short text. The `axis` column shows your exploration breadth at a glance.

## Baseline
Iteration 1 = the seed `train.py` AS IS, run once, status `keep`. Change nothing before it is recorded.

## Crashes
Trivial (typo / import) -> fix and re-run. Fundamentally broken -> log `crash`, move on. NaN pooled
stats are common -- clamp variances, avoid dividing by zero atom counts. Line-graph / three-body code is
crash-prone; build incrementally, keep `FEAT_KEY` in sync.
