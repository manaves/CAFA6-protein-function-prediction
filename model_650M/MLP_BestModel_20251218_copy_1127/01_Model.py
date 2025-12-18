import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from utils import *

# ==========================================
# 1. CAFA EVALUATION CORE (Ontology & Metrics)
# ==========================================

@dataclass
class Ont:
    ns: str
    term2i: dict[str, int]
    alt2terms: dict[str, list[str]]
    parents: list[list[int]]
    children: list[list[int]]
    nodes_by_level: list[np.ndarray]
    edge_child: list[np.ndarray]
    edge_parent_pos: list[np.ndarray]
    toi: np.ndarray
    ia: Optional[np.ndarray]
    toi_ia: Optional[np.ndarray]

@dataclass
class OntDevice:
    ns: str
    nodes_by_level: list[torch.Tensor]
    edge_child: list[torch.Tensor]
    edge_parent_pos: list[torch.Tensor]
    toi: torch.Tensor
    toi_ia: Optional[torch.Tensor]
    ia: Optional[torch.Tensor]

def _parse_ia(path: Path) -> dict[str, float]:
    d = {}
    if not path.exists(): return d
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: d[parts[0]] = float(parts[1])
    return d

def _parse_obo(obo_file: Path, ia_file: Optional[Path], orphans: bool) -> dict[str, Ont]:
    ia_dict = _parse_ia(ia_file) if ia_file else None
    term_dict = {}
    term_id, namespace, alt_id, rel, obsolete = None, None, [], [], True

    with obo_file.open() as f:
        for raw in f:
            line = raw.strip().split(": ")
            if not line or len(line) <= 1: continue
            k, v = line[0], ": ".join(line[1:])
            if k == "id":
                if term_id and not obsolete and namespace:
                    term_dict.setdefault(namespace, {})[term_id] = {"alt_id": alt_id, "rel": rel}
                term_id, namespace, alt_id, rel, obsolete = v, None, [], [], False
            elif k == "namespace": namespace = v
            elif k == "alt_id": alt_id.append(v)
            elif k == "is_obsolete": obsolete = True
            elif k == "is_a": rel.append(v.split("!")[0].strip())
            elif k == "relationship" and v.startswith("part_of"): rel.append(v.split()[1].strip())

    onts = {}
    for ns, td in term_dict.items():
        term_ids = list(td.keys())
        term2i = {t: i for i, t in enumerate(term_ids)}
        alt2terms, parents, children = {}, [[] for _ in term_ids], [[] for _ in term_ids]
        for t, rec in td.items():
            i = term2i[t]
            for a in rec["alt_id"]: alt2terms.setdefault(a, []).append(t)
            for p in rec["rel"]:
                if p in term2i:
                    j = term2i[p]
                    parents[i].append(j)
                    children[j].append(i)

        in_degree = np.array([len(c) for c in children], dtype=np.int32)
        queue = np.nonzero(in_degree == 0)[0].tolist()
        order, visited = [], 0
        while queue:
            visited += 1
            idx = queue.pop(0)
            order.append(idx)
            for j in parents[idx]:
                in_degree[j] -= 1
                if in_degree[j] == 0: queue.append(j)

        level = np.zeros(len(term_ids), dtype=np.int32)
        for i in order:
            if children[i]: level[i] = 1 + int(level[np.array(children[i])].max())

        max_lvl = int(level.max())
        nodes_by_level, edge_child, edge_parent_pos = [], [], []
        for L in range(max_lvl + 1):
            p_L = np.nonzero(level == L)[0].astype(np.int32)
            nodes_by_level.append(p_L)
            pos = {int(p): i for i, p in enumerate(p_L.tolist())}
            ch_acc, ppos_acc = [], []
            for p in p_L.tolist():
                for c in children[p]:
                    ch_acc.append(int(c)); ppos_acc.append(pos[int(p)])
            edge_child.append(np.array(ch_acc, dtype=np.int32))
            edge_parent_pos.append(np.array(ppos_acc, dtype=np.int32))

        toi = np.arange(len(term_ids)) if orphans else np.array([i for i in range(len(term_ids)) if parents[i]], dtype=np.int32)
        ia, toi_ia = None, None
        if ia_dict:
            ia = np.zeros(len(term_ids), dtype=np.float64)
            for t, i in term2i.items(): ia[i] = ia_dict.get(t, 0.0)
            toi_ia = np.nonzero(ia > 0)[0].astype(np.int32)

        onts[ns] = Ont(ns, term2i, alt2terms, parents, children, nodes_by_level, edge_child, edge_parent_pos, toi, ia, toi_ia)
    return onts

def _ont_to_device(ont: Ont, device: torch.device) -> OntDevice:
    return OntDevice(
        ns=ont.ns,
        nodes_by_level=[torch.from_numpy(a).to(device, dtype=torch.long) for a in ont.nodes_by_level],
        edge_child=[torch.from_numpy(a).to(device, dtype=torch.long) for a in ont.edge_child],
        edge_parent_pos=[torch.from_numpy(a).to(device, dtype=torch.long) for a in ont.edge_parent_pos],
        toi=torch.from_numpy(ont.toi).to(device, dtype=torch.long),
        toi_ia=torch.from_numpy(ont.toi_ia).to(device, dtype=torch.long) if ont.toi_ia is not None else None,
        ia=torch.from_numpy(ont.ia).to(device, dtype=torch.float64) if ont.ia is not None else None
    )

def _propagate_scores_levels_dev(mat: torch.Tensor, ont: OntDevice, mode: str = "fill") -> None:
    for L in range(1, len(ont.nodes_by_level)):
        p_t = ont.nodes_by_level[L]
        if p_t.numel() == 0: continue
        ec, ep = ont.edge_child[L], ont.edge_parent_pos[L]
        if ec.numel() == 0: continue
        src = mat.index_select(1, ec)
        out = torch.zeros((mat.shape[0], p_t.numel()), dtype=mat.dtype, device=mat.device)
        out.index_reduce_(1, ep, src, reduce="amax")
        if mode == "max": mat[:, p_t] = torch.maximum(mat[:, p_t], out)
        else: mat[:, p_t] = torch.where(mat[:, p_t] == 0, out, mat[:, p_t])

def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, tau: torch.Tensor, toi: torch.Tensor, ic: Optional[torch.Tensor]) -> torch.Tensor:
    gt_f, pred_f = gt[:, toi][gt[:, toi].any(dim=1)], pred[gt[:, toi].any(dim=1)][:, toi]
    P, T = pred_f.shape
    if P == 0: return torch.zeros((tau.numel(), 6), dtype=torch.float64, device=pred.device)
    
    w_vec = ic[toi] if ic is not None else None
    n_gt = (gt_f.to(torch.float64) * w_vec).sum(dim=1) if w_vec is not None else gt_f.sum(dim=1).to(torch.float64)
    
    nz = pred_f.nonzero()
    if nz.numel() == 0: return torch.zeros((tau.numel(), 6), dtype=torch.float64, device=pred.device)
    
    scores = pred_f[nz[:, 0], nz[:, 1]]
    k = torch.bucketize(scores, tau, right=True)
    keep = k > 0
    nz, k = nz[keep], k[keep]
    
    b_pred = torch.zeros(P * (tau.numel() + 1), dtype=torch.float64, device=pred.device)
    b_tp = torch.zeros_like(b_pred)
    w_pred = w_vec.index_select(0, nz[:, 1]) if w_vec is not None else torch.ones_like(k, dtype=torch.float64)
    
    flat = nz[:, 0] * (tau.numel() + 1) + k
    b_pred.scatter_add_(0, flat, w_pred)
    b_tp.scatter_add_(0, flat, w_pred * gt_f[nz[:, 0], nz[:, 1]].to(torch.float64))
    
    pred_by_tau = torch.flip(torch.cumsum(torch.flip(b_pred.view(P, -1)[:, 1:], [1]), 1), [1])
    tp_by_tau = torch.flip(torch.cumsum(torch.flip(b_tp.view(P, -1)[:, 1:], [1]), 1), [1])
    
    return torch.stack([
        (pred_by_tau > 0).sum(0).to(torch.float64), tp_by_tau.sum(0), 
        (pred_by_tau - tp_by_tau).sum(0), (n_gt.unsqueeze(1) - tp_by_tau).sum(0),
        torch.where(pred_by_tau > 0, tp_by_tau / pred_by_tau, 0.0).sum(0),
        torch.where(n_gt.unsqueeze(1) > 0, tp_by_tau / n_gt.unsqueeze(1), 0.0).sum(0)
    ], dim=1)

def _normalize(m: torch.Tensor, ns: str, tau: np.ndarray, ne: float) -> pl.DataFrame:
    arr = m.cpu().numpy()
    df = pl.DataFrame({"tau": tau, "pr": arr[:, 4]/arr[:, 0], "rc": arr[:, 5]/ne})
    df = df.with_columns(f = (2 * df["pr"] * df["rc"]) / (df["pr"] + df["rc"]).replace(0, np.nan))
    return df.fill_nan(0)

# ==========================================
# 2. MODEL & DATASET DEFINITIONS
# ==========================================

class ResidualMLP(nn.Module):
    def __init__(self, dims, dropout=0.3, output_dim=1000):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        return self.output(self.main(x))

# class SparseLabelDataset(Dataset):
#     def __init__(self, x, y):
#         self.x = torch.tensor(x, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32) # Assuming y is dense or converted
#     def __len__(self): return len(self.x)
#     def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. MAIN TRAINING & VALIDATION LOGIC
# ==========================================

class CafaValidator:
    def __init__(self, obo_path, ia_path, label_list, device):
        self.device = device
        self.onts = _parse_obo(Path(obo_path), Path(ia_path) if ia_path else None, orphans=False)
        self.onts_dev = {ns: _ont_to_device(ont, device) for ns, ont in self.onts.items()}
        self.mappings = {}
        for ns, ont in self.onts.items():
            idx_m, idx_o = [], []
            for i, go_id in enumerate(label_list):
                if go_id in ont.term2i:
                    idx_m.append(i); idx_o.append(ont.term2i[go_id])
            self.mappings[ns] = {'m': torch.tensor(idx_m, device=device), 'o': torch.tensor(idx_o, device=device)}

    def evaluate(self, all_preds, all_labels):
        results = {}
        tau = np.arange(0.01, 1.0, 0.01)
        tau_t = torch.tensor(tau, device=self.device, dtype=torch.float64)
        for ns, m in self.mappings.items():
            if m['m'].numel() == 0: continue
            p_mat = torch.zeros((all_preds.shape[0], len(self.onts[ns].term2i)), device=self.device, dtype=torch.float64)
            g_mat = torch.zeros_like(p_mat, dtype=torch.bool)
            p_mat[:, m['o']] = all_preds[:, m['m']].to(torch.float64)
            g_mat[:, m['o']] = all_labels[:, m['m']].bool()

            _propagate_scores_levels_dev(p_mat, self.onts_dev[ns], mode="fill")
            toi = self.onts_dev[ns].toi
            ne = float(g_mat[:, toi].any(1).sum().item())
            if ne == 0: continue
            
            df = _normalize(_compute_metrics(p_mat, g_mat, tau_t, toi, None), ns, tau, ne)
            results[f"{ns}_fmax"] = df["f"].max()
        return results

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mock data setup - Replace with your actual data loading
    # go_terms must be the list of GO IDs matching your label columns
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT = load_and_preprocess_data()
    N_train = len(X_train)
    
    
    go_terms = np.load(Config.CLASS_PATH)
    pos_weight_np = calculate_pos_weight(Y_train_sparse, N_train)
    pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)
    
    # Placeholder for configuration
    BATCH_SIZE, EPOCHS = Config.BATCH_SIZE, Config.NUM_EPOCHS
    DIMENSIONS = [INPUT_DIM] + Config.HIDDEN_DIMS
    
    model = ResidualMLP(DIMENSIONS, dropout=Config.DROPOUT_RATE, output_dim=LABEL_COUNT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)
    
    # Initialize CAFA Validator (Adjust paths to your local GO files)
    validator = CafaValidator(Config.OBO_FILE, Config.IA_FILE, go_terms, device)
    
    train_ds = SparseLabelDataset(X_train, Y_train_sparse)
    val_ds = SparseLabelDataset(X_val, Y_val_sparse)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation with CAFA Propagation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(torch.sigmoid(model(xb.to(device))))
                targets.append(yb.to(device))
        
        metrics = validator.evaluate(torch.cat(preds), torch.cat(targets))
        print(f"Epoch {epoch} | MFO Fmax: {metrics.get('molecular_function_fmax', 0):.4f}")

if __name__ == "__main__":
    train()