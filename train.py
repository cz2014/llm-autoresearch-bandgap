"""train.py -- THE FILE YOU EDIT. EVERYTHING here is fair game, INCLUDING the
FEATURIZATION: how a pymatgen Structure becomes model input (cutoff, connectivity / neighbor
scheme, line graphs and bond ANGLES, node/edge/global features, even multiple graphs per crystal).
The model, optimizer, schedule and training loop are equally editable. This is an algorithm +
representation search, not a hyperparameter dial.

Fixed contract (READ-ONLY prepare.py; do NOT edit it, do NOT read data/* directly):
  - prepare.train_set()              -> (list[Structure], labels[n])  official fold TRAIN
  - prepare.test_structures()        -> list[Structure]               TEST inputs (no labels)
  - prepare.evaluate_mae(predict_fn) -> MAE on held-out TEST (the ONLY score)
  - prepare.cache_dir(key) / prepare.FOLD / prepare.TIME_BUDGET / prepare.SMOKE
Goal: lowest `mae:` below. Envelope: <= 5M params (printed), single GPU, FP16 ok.

SCHEDULE NOTE: T_max is coupled to the budget's step count (SCHED_TMAX). A hardcoded constant makes
CosineAnnealingLR ramp the LR back UP once you pass it at a longer budget -- keep it coupled.
"""
import os, copy, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pymatgen.core.periodic_table import Element
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import collate_fn_graph
from matgl.graph._compute import compute_pair_vector_and_distance
import prepare

DEVICE = "cuda"
BATCH = 64

# Fixed RNG seed -> runs are reproducible, so a SINGLE run is a fair comparison between experiments
# (initialization variance, not the metric, was the main source of run-to-run noise). Keep it fixed.
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===== FEATURIZATION (agent-editable; the experiment surface) ================
CUTOFF = 5.0
FEAT_KEY = f"s2g_cut{CUTOFF}"   # cache key -- CHANGE whenever featurize() output changes

_train_structs, _train_labels = prepare.train_set()
_test_structs = prepare.test_structures()
# Fixed element vocabulary from INPUTS (train+test); element identity, not labels -> no leakage.
ELEMENTS = get_element_list(_train_structs + _test_structs)
_converter = Structure2Graph(element_types=ELEMENTS, cutoff=CUTOFF)


def _featurize_one(s):
    g, lat, state = _converter.get_graph(s)
    return g, lat, torch.as_tensor(state, dtype=torch.float32)


def featurize(structures, tag):
    """list[Structure] -> list of (graph, lattice, state) tuples, cached per (FEAT_KEY, tag, fold).
    Replace freely: different cutoff/connectivity, line graphs with angles, extra features, etc."""
    smk = f"_smoke{prepare.SMOKE}" if prepare.SMOKE else ""
    path = os.path.join(prepare.cache_dir(FEAT_KEY), f"{tag}_fold{prepare.FOLD}{smk}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    feats = [_featurize_one(s) for s in structures]
    torch.save(feats, path)
    return feats


def bond_geometry(g, lat):
    """Periodic bond distances [E] via matgl's tested PBC routine (moved here from prepare.py --
    geometry IS featurization now; extend it, e.g. also return angles, if you change the graph)."""
    ei = g.edge_index.long()
    batch = getattr(g, "batch", None)
    if batch is None:
        L = lat[0]
        pos = g.frac_coords @ L
        off = g.pbc_offset @ L
    else:
        Ln = lat[batch]
        pos = torch.bmm(g.frac_coords.unsqueeze(1), Ln).squeeze(1)
        Le = lat[batch[ei[0]]]
        off = torch.bmm(g.pbc_offset.unsqueeze(1), Le).squeeze(1)
    _, dist = compute_pair_vector_and_distance(pos, ei, off)
    return dist


def _loader(feats, labels, shuffle):
    items = [(*feats[i], {"gap": float(labels[i] if labels is not None else 0.0)})
             for i in range(len(feats))]
    return torch.utils.data.DataLoader(items, batch_size=BATCH, shuffle=shuffle,
                                       collate_fn=collate_fn_graph)


# ===== MODEL (round-1 winner 66d5f2a; ~103k params) -- edit me ===============
def _el_scalar(el, name, scale, default=0.0):
    v = getattr(el, name, None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = default
    return (v if v is not None else default) / scale


def element_feature_table():
    rows = []
    for sym in ELEMENTS:
        el = Element(str(sym))
        rows.append([
            _el_scalar(el, "Z", 100.0), _el_scalar(el, "row", 7.0), _el_scalar(el, "group", 18.0),
            _el_scalar(el, "X", 4.0), _el_scalar(el, "atomic_radius", 3.0),
            _el_scalar(el, "atomic_radius_calculated", 3.0),
            _el_scalar(el, "van_der_waals_radius", 3.0), _el_scalar(el, "mendeleev_no", 103.0),
        ])
    f = torch.tensor(rows, dtype=torch.float32)
    return (f - f.mean(0)) / f.std(0).clamp_min(1e-6)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=32, cutoff=5.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class CGConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin_f = nn.Linear(3 * dim, dim)
        self.lin_s = nn.Linear(3 * dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, h, edge_index, e):
        src, dst = edge_index[0].long(), edge_index[1].long()
        z = torch.cat([h[dst], h[src], e], dim=-1)
        m = torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        return self.ln(h + self.bn(agg))


class CrystalGNN(nn.Module):
    def __init__(self, num_elem, dim=64, n_layers=4, n_rbf=32, cutoff=5.0):
        super().__init__()
        self.embed = nn.Embedding(num_elem, dim)
        self.register_buffer("elem_feat", element_feature_table())
        self.elem_mlp = nn.Sequential(nn.Linear(self.elem_feat.size(1), dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.edge_mlp = nn.Sequential(nn.Linear(n_rbf, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.convs = nn.ModuleList([CGConv(dim) for _ in range(n_layers)])
        self.global_mlp = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.readout = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(),
                                     nn.Dropout(0.05), nn.Linear(dim, 1))

    def global_features(self, g, lat, dist, dtype):
        counts = torch.bincount(g.batch, minlength=lat.size(0)).float().clamp_min(1.0)
        volume = torch.linalg.det(lat.float()).abs().clamp_min(1e-6)
        vpa = volume / counts
        density = counts / volume
        eb = g.batch[g.edge_index[0].long()]
        md = scatter(dist.float(), eb, dim=0, dim_size=lat.size(0), reduce="mean")
        md2 = scatter(dist.float().square(), eb, dim=0, dim_size=lat.size(0), reduce="mean")
        sd = (md2 - md.square()).clamp_min(0.0).sqrt()
        x = torch.stack([torch.log1p(counts), torch.log1p(volume), torch.log1p(vpa),
                         torch.log1p(density), torch.log1p(md), torch.log1p(sd)], dim=-1)
        return self.global_mlp(x.to(dtype=dtype))

    def forward(self, g, lat):
        dist = bond_geometry(g, lat)
        nt = g.node_type.long()
        h = self.embed(nt) + self.elem_mlp(self.elem_feat[nt])
        e = self.edge_mlp(self.rbf(dist))
        for conv in self.convs:
            h = conv(h, g.edge_index, e)
        hg = torch.cat([global_mean_pool(h, g.batch), global_max_pool(h, g.batch),
                        self.global_features(g, lat, dist, h.dtype)], dim=-1)
        return self.readout(hg).squeeze(-1)


model = CrystalGNN(num_elem=len(ELEMENTS), dim=96, n_layers=4, n_rbf=32, cutoff=CUTOFF).to(DEVICE)
ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# cosine horizon coupled to the budget (~44 steps/s on this GPU; small margin so the schedule
# completes near the budget end without the LR ramping back up -- see header note).
SCHED_TMAX = max(1000, int(48.0 * prepare.TIME_BUDGET))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SCHED_TMAX, eta_min=1e-5)
# ===========================================================================


@torch.no_grad()
def predict_fn(structures):
    """Featurize TEST inputs with the SAME featurizer and run the (EMA) model. Returns array[len]."""
    feats = featurize(structures, "test")
    loader = _loader(feats, None, shuffle=False)
    ema_model.eval()
    out = []
    for g, lat, state, _ in loader:
        g, lat = g.to(DEVICE), lat.to(DEVICE)
        out.append(ema_model(g, lat).view(-1).float().cpu())
    return torch.cat(out).numpy()


def main():
    t_feat = time.time()
    train_feats = featurize(_train_structs, "train")
    feat_secs = time.time() - t_feat
    loader = _loader(train_feats, _train_labels, shuffle=True)
    n_params = sum(p.numel() for p in model.parameters())

    t0, step = time.time(), 0
    model.train()
    while time.time() - t0 < prepare.TIME_BUDGET:
        for g, lat, state, labels in loader:
            g, lat, labels = g.to(DEVICE), lat.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(g, lat).view(-1)
                loss = F.l1_loss(preds, labels.view(-1).float())
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                for ep, p in zip(ema_model.parameters(), model.parameters()):
                    ep.lerp_(p, 0.005)
                for eb, b in zip(ema_model.buffers(), model.buffers()):
                    eb.copy_(b)
            if time.time() - t0 >= prepare.TIME_BUDGET:
                break
    train_secs = time.time() - t0

    mae = prepare.evaluate_mae(predict_fn)
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print("---")
    print(f"mae:              {mae:.6f}")
    print(f"training_seconds: {train_secs:.1f}")
    print(f"feat_seconds:     {feat_secs:.1f}")
    print(f"peak_vram_mb:     {peak:.1f}")
    print(f"num_params:       {n_params}")
    print(f"steps:            {step}")


if __name__ == "__main__":
    main()
