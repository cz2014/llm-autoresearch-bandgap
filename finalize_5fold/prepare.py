"""prepare.py -- READ-ONLY. The agent edits ONLY train.py and NEVER this file.

FROZEN here: the raw data, the official fold split, the held-out TEST labels, and the MAE metric.
NOT here: featurization. The agent builds the crystal -> model-input featurization INSIDE train.py
(cutoff, connectivity, line graphs/angles, node/edge/global features), so this file is
graph-agnostic: it hands out pymatgen Structures and takes back predictions.

Contract train.py must satisfy:
  - prepare.train_set()              -> (list[Structure], np.ndarray[n_train])  official fold TRAIN
  - prepare.test_structures()        -> list[Structure]                          TEST inputs, NO labels
  - prepare.evaluate_mae(predict_fn) -> float
        predict_fn(list[Structure]) -> array[len] of predicted gaps; MAE vs the private TEST labels.
  - prepare.cache_dir(key)           -> a per-featurization cache dir (data/graphs/<key>/)
  - prepare.FOLD / prepare.TIME_BUDGET / prepare.SMOKE

The ONLY path to a score is evaluate_mae; TEST labels never leave this module. train.py must obtain
data ONLY through these functions -- reading data/*.json* directly is forbidden (audited in git).

Ops knobs: MPGAP_FOLD (default 0), MPGAP_TIME_BUDGET seconds (default 1500), MPGAP_SMOKE (default 0
= full; >0 subsamples TRAIN to that many and TEST to a quarter, for fast pipeline tests).
"""
import os, json, gzip
import numpy as np
from pymatgen.core import Structure

FOLD = int(os.environ.get("MPGAP_FOLD", "0"))
TIME_BUDGET = float(os.environ.get("MPGAP_TIME_BUDGET", "1500"))
SMOKE = int(os.environ.get("MPGAP_SMOKE", "0"))
_DATA, _FOLDS = "data/mp_gap.json.gz", "data/folds.json"
_cache = {}


def _load():
    if _cache:
        return _cache
    with gzip.open(_DATA, "rt") as f:
        d = json.load(f)
    structures = [Structure.from_dict(s) for s in d["structures"]]
    gaps = np.asarray([float(x) for x in d["gaps"]], dtype=np.float64)
    meta = json.load(open(_FOLDS))
    assert meta["n"] == len(structures), (
        f"folds.json n={meta['n']} != {len(structures)} structures; re-run make_data.py")
    fold = meta["folds"][FOLD]
    train_idx, test_idx = list(fold["train"]), list(fold["test"])
    if SMOKE:
        train_idx = train_idx[:SMOKE]
        test_idx = test_idx[:max(1, SMOKE // 4)]
    _cache.update(structures=structures, gaps=gaps, train_idx=train_idx, test_idx=test_idx)
    return _cache


def train_set():
    """(list[Structure], np.ndarray[n_train]) for the official fold TRAIN split."""
    c = _load()
    return [c["structures"][i] for i in c["train_idx"]], c["gaps"][c["train_idx"]]


def test_structures():
    """list[Structure] for the held-out TEST split (inputs only -- labels never leave here)."""
    c = _load()
    return [c["structures"][i] for i in c["test_idx"]]


def _test_labels():
    c = _load()
    return c["gaps"][c["test_idx"]]


def evaluate_mae(predict_fn):
    """GROUND-TRUTH metric. predict_fn(test_structures()) -> array[len(test)] predicted gaps.
    Returns MAE (eV) vs the privately held TEST labels."""
    labels = _test_labels()
    preds = np.asarray(predict_fn(test_structures()), dtype=np.float64).reshape(-1)
    assert preds.shape == labels.shape, (
        f"predict_fn returned shape {preds.shape}, expected {labels.shape}")
    return float(np.abs(preds - labels).mean())


def cache_dir(key):
    """Per-featurization cache dir: data/graphs/<sanitized key>/. The agent picks the key
    (e.g. 's2g_cut5.0') and MUST change it whenever featurize() output changes."""
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(key))
    path = os.path.join("data", "graphs", safe)
    os.makedirs(path, exist_ok=True)
    return path
