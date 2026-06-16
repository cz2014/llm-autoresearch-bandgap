"""make_data.py -- run ONCE under the matminer env (.venv_data, py3.12).
Produces the fixed inputs the substrate consumes:
  data/mp_gap.json.gz : {"structures":[pymatgen as_dict...], "gaps":[float...], "n":N}
  data/folds.json     : official MatBench v0.1 split, reproduced with
                        KFold(n_splits=5, shuffle=True, random_state=18012019)
                        (regression seed from ref/matbench scripts/mvb01_generate_validation.py).
Set MPGAP_LIMIT=N for a small smoke dataset.
"""
import os, json, gzip
import numpy as np
from sklearn.model_selection import KFold
from matminer.datasets import load_dataset

LIMIT = int(os.environ.get("MPGAP_LIMIT", "0")) or None
os.makedirs("data", exist_ok=True)
df = load_dataset("matbench_mp_gap")          # canonical order; columns: 'structure','gap pbe'
if LIMIT:
    df = df.iloc[:LIMIT].copy()
N = len(df)
structs = [s.as_dict() for s in df["structure"]]
gaps = [float(x) for x in df["gap pbe"]]
with gzip.open("data/mp_gap.json.gz", "wt") as f:
    json.dump({"structures": structs, "gaps": gaps, "n": N}, f)

kf = KFold(n_splits=5, shuffle=True, random_state=18012019)
folds = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in kf.split(np.arange(N))]
json.dump({"n_splits": 5, "random_state": 18012019, "shuffle": True,
           "official_seed_reproduced": True, "n": N, "folds": folds}, open("data/folds.json", "w"))
print(f"wrote data/mp_gap.json.gz (N={N}); data/folds.json "
      f"(fold0 train={len(folds[0]['train'])} test={len(folds[0]['test'])})")
