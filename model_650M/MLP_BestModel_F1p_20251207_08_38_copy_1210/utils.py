import os
import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import networkx as nx
import obonet
import sys
import logging
from torch.utils.data import Dataset
from tqdm import tqdm
from config import Config

# --- UTILITY FUNCTIONS ---

def set_seed(seed=Config.SEED):
    """Sets a fixed seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalizes each row of a numpy array to have a unit L2 norm."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 
    return x / norms

def safe_load_sparse_array(path: str):
    """Loads a numpy array that may contain a sparse matrix object."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    return obj

def apply_label_smoothing(targets: torch.Tensor, eps: float) -> torch.Tensor:
    """Applies label smoothing to target tensor."""
    if eps <= 0.0:
        return targets
    return targets * (1.0 - eps) + 0.5 * eps

def calculate_pos_weight(Y_train, N: int, eps: float = 1e-6) -> np.ndarray:
    """Calculates positive weight for BCEWithLogitsLoss based on class frequency."""
    logging.info("Calculating pos_weight per label...")
    
    if sp.issparse(Y_train):
        label_freq = np.asarray(Y_train.sum(axis=0)).ravel()
    else:
        label_freq = np.sum(Y_train, axis=0)

    label_freq = label_freq.astype(np.float32)
    label_freq[label_freq < 1.0] = 1.0
    
    pos_weight = (N - label_freq) / (label_freq + eps)
    
    return np.clip(pos_weight, a_min=1.0, a_max=100.0)

# --- DATA LOADING AND PREPARATION ---

def load_and_preprocess_data():
    """Loads, normalizes data, and determines input/label dimensions."""
    logging.info("Loading embeddings and labels...")
    
    X_train = np.load(Config.TRAIN_EMB_PATH)
    X_val = np.load(Config.VAL_EMB_PATH)

    Y_train_sparse = safe_load_sparse_array(Config.Y_TRAIN_PATH)
    Y_val_sparse = safe_load_sparse_array(Config.Y_VAL_PATH)

    X_train = normalize_rows(X_train)
    X_val = normalize_rows(X_val)

    input_dim = X_train.shape[1]
    
    if sp.issparse(Y_train_sparse):
        label_count = Y_train_sparse.shape[1]
    else:
        label_count = np.array(Y_train_sparse).shape[1]

    return X_train, X_val, Y_train_sparse, Y_val_sparse, input_dim, label_count

# --- GO PROPAGATION UTILITIES ---

def load_go_dag(obo_path: str) -> nx.DiGraph:
    """Reads the Gene Ontology OBO file and converts it to a NetworkX DiGraph."""
    logging.info("1. Reading GO file and building DAG...")
    try:
        graph = obonet.read_obo(obo_path)
    except FileNotFoundError:
        logging.error(f"OBO file not found at: {obo_path}. Check Config.OBO_FILE_PATH.")
        sys.exit(1)
    
    # Initialize a new DiGraph for propagation: Child -> Parent (Successor is Parent)
    go_dag = nx.DiGraph()
    
    for node, data in graph.nodes(data=True):
        go_dag.add_node(node)
        # 'is_a' relationships define parent terms
        if 'is_a' in data:
            for parent in data['is_a']:
                # Edge: Child -> Parent (for upward propagation)
                go_dag.add_edge(node, parent)
                
    logging.info(f"Graph loaded: {go_dag.number_of_nodes()} nodes.")
    return go_dag

def map_model_classes(classes_path: str, go_dag: nx.DiGraph) -> tuple[dict, list]:
    """Loads model class IDs and maps GO terms to their column indices."""
    logging.info("2. Mapping classes from the model to GO terms...")
    try:
        model_classes = np.load(classes_path)
    except FileNotFoundError:
        logging.error(f"Classes file not found at: {classes_path}. Check Config.CLASSES_PATH.")
        sys.exit(1)
        
    # Mapping GO Term ID -> Column Index
    class_to_idx = {term: i for i, term in enumerate(model_classes)}

    # Identify which terms from the model are actually in the GO DAG
    valid_terms = [term for term in model_classes if term in go_dag]
    
    logging.info(f"Valid terms found in DAG: {len(valid_terms)} out of {len(model_classes)}")
    return class_to_idx, model_classes

def get_propagation_steps(go_dag: nx.DiGraph, class_to_idx: dict) -> list[tuple[int, int]]:
    """
    Determines the ordered steps (child_idx, parent_idx) needed for score propagation.
    The order ensures child scores are processed before their parents.
    """
    logging.info("3. Establishing topological order for propagation...")
    
    # Topological sort (Child -> Parent): ensures that children are processed before parents
    try:
        # Full topological order of all nodes in the DAG
        full_topological_order = list(nx.topological_sort(go_dag))
    except nx.NetworkXUnfeasible:
        # Should not happen in GO DAG, but good practice to catch cycles.
        logging.error("GO DAG contains a cycle, cannot perform topological sort.")
        sys.exit(1)

    # We want to iterate from children (leaf terms) up to parents (root terms).
    # Since topological_sort orders from root to leaf, we reverse it.
    sorted_model_terms = [term for term in reversed(full_topological_order) if term in class_to_idx]

    propagation_steps = []
    
    logging.info("Generating propagation steps...")
    
    for child in sorted_model_terms:
        child_idx = class_to_idx[child]
        
        # Successors in a Child->Parent graph are the direct parents
        if child in go_dag:
            parents = list(go_dag.successors(child)) 
            for parent in parents:
                if parent in class_to_idx:
                    parent_idx = class_to_idx[parent]
                    # Store the indices: (Child_Index, Parent_Index)
                    propagation_steps.append((child_idx, parent_idx))

    logging.info(f"Parent-child relationships to be processed: {len(propagation_steps)}")
    return propagation_steps

# --- PYTORCH DATASET AND MODEL DEFINITION ---

class SparseLabelDataset(Dataset):
    """Dataset class for handling embeddings and potentially sparse labels."""
    def __init__(self, X: np.ndarray, Y_sparse):
        self.X = X.astype(np.float32)
        self.Y = Y_sparse

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        
        if sp.issparse(self.Y):
            y = torch.from_numpy(self.Y[idx].toarray().reshape(-1).astype(np.float32))
        else:
            y = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32).reshape(-1))
            
        return x, y

# --- MODEL ARCHITECTURE (Residual MLP) ---

class ResidualBlock(nn.Module):
    """A compact residual block with Linear-LayerNorm-GELU-Dropout."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies residual connection: output = x + block(x)"""
        return x + self.block(x)
    
class ResidualMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) with a residual architecture."""
    def __init__(self, dims: list, dropout: float, output_dim: int):
        super().__init__()
        layers = []
        input_dim = dims[0]
        
        layers.append(nn.Linear(input_dim, dims[1]))
        layers.append(nn.LayerNorm(dims[1]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        for i in range(1, len(dims)-1):
            dim_in = dims[i]
            dim_out = dims[i+1]
            
            if dim_in == dim_out:
                layers.append(ResidualBlock(dim_in, dropout))
            else:
                layers.append(nn.Linear(dim_in, dim_out))
                layers.append(nn.LayerNorm(dim_out))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits: head(encoder(x))"""
        encoded_x = self.encoder(x)
        logits = self.head(encoded_x)
        return logits

# --- VALIDATION FUNCTIONS ---

def validate_cafa_pk(
    model: nn.Module,
    loader,
    ia_weights_np: np.ndarray,
    propagation_steps: list[tuple[int, int]],
    label_count: int,
    device,
    th_step: float = 0.05):
    """
    Official CAFA-PK evaluation (Protein-centric F-max).
    """
    model.eval()

    # Optimized range for speed during training
    # We search between 0.0 and 1.0. 
    # Note: Rarely the optimal threshold is at the extremes 0.0 or 1.0 in CAFA.
    thresholds = np.arange(0.1, 0.95, th_step) 
    T = len(thresholds)

    sum_precision = np.zeros(T, dtype=np.float64)
    sum_recall = np.zeros(T, dtype=np.float64)
    
    num_proteins = 0
    nan_detected = False

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            if torch.isnan(logits).any():
                nan_detected = True; break

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = yb.cpu().numpy().astype(np.int8)

            # 1. Propagation
            for child_idx, parent_idx in propagation_steps:
                probs[:, parent_idx] = np.maximum(probs[:, parent_idx], probs[:, child_idx])

            batch_size = probs.shape[0]
            num_proteins += batch_size

            # 2. Metrics by protein (Vectorized for speed)
            # Pre-multiplying targets and probs by IA weights saves time inside the thresholds loop
            # But as the preds change with the threshold, we do it partially inside.
            
            for ti, t in enumerate(thresholds):
                preds = (probs >= t).astype(np.int8)

                # Calculate intersection (TP) and weighted unions
                # TP = preds & targets
                # FP = preds & ~targets
                # FN = ~preds & targets
                
                # Optimization trick: 
                # P = (TP_w) / (TP_w + FP_w) = (preds * targets * w).sum() / (preds * w).sum()
                # R = (TP_w) / (TP_w + FN_w) = (preds * targets * w).sum() / (targets * w).sum()
                
                # Weights for this batch
                # shape: [batch, labels]
                
                # Numerator (Weighted TP)
                tp_w = (preds * targets * ia_weights_np).sum(axis=1)
                
                # Denominator Precision (All predicted weighted)
                pred_w = (preds * ia_weights_np).sum(axis=1)
                
                # Denominator Recall (All real targets weighted - constant for this batch, but depends on the target)
                target_w = (targets * ia_weights_np).sum(axis=1)

                # Prevent division by zero
                prec = np.divide(tp_w, pred_w, out=np.zeros_like(tp_w), where=pred_w > 0)
                rec  = np.divide(tp_w, target_w, out=np.zeros_like(tp_w), where=target_w > 0)

                sum_precision[ti] += prec.sum()
                sum_recall[ti]    += rec.sum()

    if nan_detected: raise RuntimeError("NaN in logits")

    # 3. Average Precision and Recall
    avg_precision = sum_precision / num_proteins
    avg_recall = sum_recall / num_proteins

    # F-score
    denom = avg_precision + avg_recall
    f_scores = np.divide(2 * avg_precision * avg_recall, denom, out=np.zeros_like(avg_precision), where=denom > 0)

    best_idx = int(np.argmax(f_scores))
    return float(thresholds[best_idx]), float(f_scores[best_idx])

def validate_epoch_propagated(model: nn.Module, loader, label_count: int, 
                              ia_weights_np: np.ndarray, 
                              propagation_steps: list[tuple[int, int]]) -> tuple:
    """
    Performs validation, applies GO propagation to predictions, and computes the 
    best WEIGHTED F1-score.
    """
    model.eval()
    
    thresholds = np.arange(0.35, 0.95, 0.05)
    T = len(thresholds)

    # Initialize per-threshold, per-label counters
    TP = np.zeros((T, label_count), dtype=np.float64)
    FP = np.zeros((T, label_count), dtype=np.float64)
    FN = np.zeros((T, label_count), dtype=np.float64)
    
    nan_detected = False

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Validation"):
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)

            # 1. Get Logits
            if Config.USE_AMP and Config.DEVICE.type == "cuda":
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_detected = True
                break

            # 2. Convert to Probabilities (scores) and Numpy
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = yb.cpu().numpy().astype(np.int8) 

            # 3. APPLY GO PROPAGATION (Maximum Rule: Parent_score = max(Parent_score, Child_score))
            for child_idx, parent_idx in propagation_steps:
                probs[:, parent_idx] = np.maximum(
                    probs[:, parent_idx], 
                    probs[:, child_idx]
                )

            # 4. Calculate Metrics for each threshold (on PROPAGATED probabilities)
            for i, t in enumerate(thresholds):
                preds = (probs >= t).astype(np.int8)
                
                # Calculate True Positives, False Positives, False Negatives for the batch
                TP_batch = (preds == 1) & (targets == 1)
                FP_batch = (preds == 1) & (targets == 0)
                FN_batch = (preds == 0) & (targets == 1)
                
                # Accumulate the weighted counts (multiply by IA weights)
                TP[i] += np.sum(TP_batch * ia_weights_np, axis=0)
                FP[i] += np.sum(FP_batch * ia_weights_np, axis=0)
                FN[i] += np.sum(FN_batch * ia_weights_np, axis=0)

    if nan_detected:
        raise RuntimeError("Validation aborted due to NaN/Inf in logits.")

    # --- FINAL WEIGHTED F1 CALCULATION ---
    weighted_f1_per_threshold = np.zeros(T)
    
    TP_sum = np.sum(TP, axis=1) 
    FP_sum = np.sum(FP, axis=1)
    FN_sum = np.sum(FN, axis=1)
    
    for i in range(T):
        tp, fp, fn = TP_sum[i], FP_sum[i], FN_sum[i]
        
        P_w = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R_w = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1_w = (2 * P_w * R_w) / (P_w + R_w) if (P_w + R_w) > 0 else 0.0
        
        weighted_f1_per_threshold[i] = F1_w

    # Select best threshold and F1
    best_idx = int(np.argmax(weighted_f1_per_threshold))
    best_t = float(thresholds[best_idx])
    best_f1_local = float(weighted_f1_per_threshold[best_idx])
    
    return best_t, best_f1_local

