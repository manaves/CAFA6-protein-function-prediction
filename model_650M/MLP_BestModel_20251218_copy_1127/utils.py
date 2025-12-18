import torch
import torch.nn as nn

from config import TrainConfig as Config
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import time

import numpy as np
import random
import scipy.sparse as sp

import networkx as nx
import os
import obonet

import copy
from pathlib import Path

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
        """Applies residual connection: output = x + block(x)."""
        return x + self.block(x)


class ResidualMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with a residual architecture.
    Uses ResidualBlock when input_dim == output_dim.
    """

    def __init__(self, dims: list, dropout: float, output_dim: int):
        super().__init__()
        layers = []
        input_dim = dims[0]

        # Initial projection layer (dims[0] -> dims[1])
        layers.append(nn.Linear(input_dim, dims[1]))
        layers.append(nn.LayerNorm(dims[1]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for i in range(1, len(dims) - 1):
            dim_in = dims[i]
            dim_out = dims[i + 1]

            if dim_in == dim_out:
                # Use Residual Block for same dimensions
                layers.append(ResidualBlock(dim_in, dropout))
            else:
                # Use standard block for dimension change
                layers.append(nn.Linear(dim_in, dim_out))
                layers.append(nn.LayerNorm(dim_out))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], output_dim)

        # NOTE: forward is intentionally identical between training and inference.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits as head(encoder(x))."""
        encoded_x = self.encoder(x)
        logits = self.head(encoded_x)
        return logits

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
    # Avoid division by zero for zero vectors
    norms[norms == 0] = 1.0 
    return x / norms

def safe_load_sparse_array(path: str):
    """Loads a numpy array that may contain a sparse matrix object."""
    obj = np.load(path, allow_pickle=True) # Changed from sp.load_npz to np.load
    # Handles case where a sparse object is saved in a 0-dim numpy array
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
    print("Calculating pos_weight per label...")
    
    if sp.issparse(Y_train):
        # Sparse matrix sum
        label_freq = np.asarray(Y_train.sum(axis=0)).ravel()
    else:
        # Dense array sum
        label_freq = np.sum(Y_train, axis=0)

    # Convert to float for division and set minimum frequency to 1
    label_freq = label_freq.astype(np.float32)
    label_freq[label_freq < 1.0] = 1.0
    
    # pos_weight = (N - n_i) / n_i
    pos_weight = (N - label_freq) / (label_freq + eps)
    
    # Clip weights to a reasonable range
    return np.clip(pos_weight, a_min=1.0, a_max=100.0)

# --- DATA LOADING AND PREPARATION ---

def load_and_preprocess_data():
    """Loads, normalizes data, and determines input/label dimensions."""
    print("Loading embeddings and labels...")
    
    # Load features
    X_train = np.load(Config.TRAIN_EMB_PATH)
    X_val = np.load(Config.VAL_EMB_PATH)

    # Load labels
    Y_train_sparse = safe_load_sparse_array(Config.Y_TRAIN_PATH)
    Y_val_sparse = safe_load_sparse_array(Config.Y_VAL_PATH)

    # Normalization
    X_train = normalize_rows(X_train)
    X_val = normalize_rows(X_val)

    # Determine dimensions
    input_dim = X_train.shape[1]
    
    if sp.issparse(Y_train_sparse):
        label_count = Y_train_sparse.shape[1]
    else:
        label_count = np.array(Y_train_sparse).shape[1]

    print(f"Input dim: {input_dim}, labels: {label_count}")
    return X_train, X_val, Y_train_sparse, Y_val_sparse, input_dim, label_count

# --- PYTORCH DATASET AND MODEL DEFINITION ---

class SparseLabelDataset(Dataset):
    """
    Dataset class for handling embeddings and potentially sparse labels.
    Converts data to the required float32 PyTorch format.
    """
    def __init__(self, X: np.ndarray, Y_sparse):
        self.X = X.astype(np.float32)
        self.Y = Y_sparse

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        
        if sp.issparse(self.Y):
            # Convert sparse row to dense array and reshape to vector
            y = torch.from_numpy(self.Y[idx].toarray().reshape(-1).astype(np.float32))
        else:
            # Convert dense row to vector
            y = torch.from_numpy(np.array(self.Y[idx], dtype=np.float32).reshape(-1))
            
        return x, y

        # --- MODEL ARCHITECTURE (Residual MLP) is now in utils.py ---

# --- EVALUATION METRICS AND HELPERS ---

def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray, 
                        thresholds: np.ndarray = np.arange(0.05, 0.51, 0.01)) -> tuple:
    """
    Finds the best classification threshold for macro F1-score on the validation set.
    Requires scikit-learn.
    """
    if f1_score is None:
        print("sklearn missing: returning 0.5 as threshold")
        return 0.5, 0.0
        
    best_t = 0.5
    best_f1 = -1.0
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        try:
            # Macro F1 for multi-label (average over labels)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            # Fallback for older sklearn versions (average over samples)
            f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
            
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    return best_t, best_f1

# --- TRAINING AND VALIDATION LOGIC ---

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, scaler, epoch: int):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [train]", leave=False)

    for xb, yb in pbar:
        xb = xb.to(Config.DEVICE)
        yb = yb.to(Config.DEVICE)

        if Config.LABEL_SMOOTHING_EPSILON > 0:
            yb = apply_label_smoothing(yb, Config.LABEL_SMOOTHING_EPSILON)

        optimizer.zero_grad()

        # Mixed Precision Training (if enabled)
        if Config.USE_AMP and Config.DEVICE.type == "cuda":
            with torch.autocast(device_type=Config.DEVICE.type):
                logits = model(xb)
                loss = criterion(logits, yb)
                
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("NaN/Inf detected in training loss.")

            # Gradient scaling, unscaling, clipping, and step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            # Standard Training
            logits = model(xb)
            loss = criterion(logits, yb)

            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("NaN/Inf detected in training loss.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate_epoch(model: nn.Module, loader: DataLoader, label_count: int) -> tuple:
    """Performs validation and computes the best F1-score and threshold."""
    model.eval()
    
    # Thresholds for validation F1 tuning
    thresholds = np.arange(0.05, 0.51, 0.05)
    T = len(thresholds)

    # Initialize per-threshold, per-label counters for True Positives, False Positives, False Negatives
    TP = np.zeros((T, label_count), dtype=np.int64)
    FP = np.zeros((T, label_count), dtype=np.int64)
    FN = np.zeros((T, label_count), dtype=np.int64)
    
    nan_detected = False

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)

            # AMP also applies for validation
            if Config.USE_AMP and Config.DEVICE.type == "cuda":
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)

            # Check for NaN/Inf in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN/Inf detected in validation logits.")
                nan_detected = True
                break

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = yb.cpu().numpy().astype(np.int8)

            # Update metrics for each threshold
            for i, t in enumerate(thresholds):
                preds = (probs >= t).astype(np.int8)
                TP[i] += np.sum((preds == 1) & (targets == 1), axis=0)
                FP[i] += np.sum((preds == 1) & (targets == 0), axis=0)
                FN[i] += np.sum((preds == 0) & (targets == 1), axis=0)

    if nan_detected:
        raise RuntimeError("Validation aborted due to NaN/Inf in logits.")

    # Compute macro F1 per threshold
    f1_per_threshold = np.zeros(T)
    for i in range(T):
        tp = TP[i].astype(float)
        fp = FP[i].astype(float)
        fn = FN[i].astype(float)
        
        # F1_label = 2 * TP / (2 * TP + FP + FN)
        denom = 2 * tp + fp + fn
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero, set F1 to 0.0 where denominator is zero
            f1_label = np.where(denom > 0, (2 * tp) / denom, 0.0)
        
        # Macro F1 is the mean F1 across all labels
        f1_per_threshold[i] = np.mean(f1_label)

    # Select best threshold and F1
    best_idx = int(np.argmax(f1_per_threshold))
    best_t = float(thresholds[best_idx])
    best_f1_local = float(f1_per_threshold[best_idx])
    
    return best_t, best_f1_local

def save_checkpoint(model, optimizer, epoch, threshold, val_f1):
    """Saves the model state, optimizer state, and metrics."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "threshold": threshold,
        "val_f1": val_f1,
    }
    torch.save(checkpoint, Config.MODEL_SAVE_PATH)
    print("  -> New best model saved.")

def save_predictions_to_memmap(model, loader: DataLoader, total_samples: int, label_count: int):
    """Loads best model and saves its logits (predictions) to a numpy memory-mapped file."""
    
    # Load best model checkpoint
    print("Loading best model for final prediction saving...")
    ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    best_threshold = ckpt.get('threshold', 0.5)
    best_val_f1 = ckpt.get('val_f1', 0.0)
    
    # Create memory-mapped file
    print(f"Creating memmap at: {Config.PREDICTIONS_MEMMAP_PATH} with shape ({total_samples}, {label_count})")
    fp = np.memmap(
        Config.PREDICTIONS_MEMMAP_PATH, 
        dtype='float32', 
        mode='w+', 
        shape=(total_samples, label_count)
    )

    model.eval()
    start_idx = 0
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Saving predictions..."):
            xb = xb.to(Config.DEVICE)
            
            if Config.USE_AMP and Config.DEVICE.type == 'cuda':
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)
                
            arr = logits.cpu().numpy()
            batch_size = arr.shape[0]
            fp[start_idx : start_idx + batch_size, :] = arr
            start_idx += batch_size

    fp.flush() # Ensure data is written to disk
    print(f"Predictions saved in: {Config.PREDICTIONS_MEMMAP_PATH}")
    print(f"Best F1 (val): {best_val_f1:.5f} with threshold {best_threshold}")


# --- DATA PREPARATION ---

def load_data_and_model() -> tuple:
    """Loads FASTA IDs, embeddings, model, and calculates dimensions."""
    print("## Loading Data and Model")
    
    # Read FASTA IDs
    test_ids = []
    with open(Config.FASTA_PATH, 'r') as f:
        for line in tqdm(f, desc="Reading FASTA IDs"):
            if line.startswith(">"):
                test_ids.append(line.strip().split()[0][1:])

    # Load Embeddings (using mmap for efficiency)
    emb_test = np.load(Config.EMBEDDINGS_PATH, mmap_mode="r")
    real_samples = emb_test.shape[0]

    # Align samples and IDs
    NUM_SAMPLES = len(test_ids)
    if real_samples != NUM_SAMPLES:
        print(f"⚠️ WARNING: Sample count mismatch ({real_samples} in file vs {NUM_SAMPLES} in FASTA). Truncating.")
        NUM_SAMPLES = min(real_samples, NUM_SAMPLES)
        test_ids = test_ids[:NUM_SAMPLES]

    EMBEDDING_DIM = emb_test.shape[1]
    
    print(f"Total samples to process: {NUM_SAMPLES}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    # Load Model and Checkpoint
    model_classes = np.load(Config.CLASSES_PATH, allow_pickle=True)
    NUM_CLASSES = len(model_classes)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    # Model initialization
    HIDDEN_DIMS = [EMBEDDING_DIM, 1024, 1024, 512] 
    model = ResidualMLP(dims=HIDDEN_DIMS, dropout=Config.DROPOUT, output_dim=NUM_CLASSES).to(Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Update threshold from checkpoint if available
    if Config.THRESHOLD > 0.0:
        threshold = Config.THRESHOLD
    else:
        threshold = checkpoint.get("threshold", Config.THRESHOLD)
    print(f"Using prediction threshold: {threshold:.4f} (from checkpoint or config)")
    
    return model, emb_test, test_ids, model_classes, NUM_SAMPLES, NUM_CLASSES, threshold

# --- INFERENCE, PROPAGATION, AND SUBMISSION ---

def run_inference(model: nn.Module, emb_test: np.memmap, num_samples: int, num_classes: int):
    """Runs inference on test embeddings and saves raw probabilities to disk."""
    print("## Running Inference")
    
    fp_raw = np.memmap(
        Config.RAW_PREDS_PATH, 
        dtype="float32", 
        mode="w+", 
        shape=(num_samples, num_classes)
    )

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, Config.BATCH_SIZE), desc="Predicting"):
            end = min(i + Config.BATCH_SIZE, num_samples)

            # Load batch from memmap and move to device
            batch = torch.tensor(emb_test[i:end], dtype=torch.float32).to(Config.DEVICE)
            
            # Forward pass: Logits -> Sigmoid -> Probabilities
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Save probabilities to raw memmap
            fp_raw[i:end] = probs

            # Flush periodically
            if i % (Config.BATCH_SIZE * 5) == 0:
                fp_raw.flush()
                time.sleep(0.05)

    fp_raw.flush()
    print(f"Raw inference predictions saved to: {Config.RAW_PREDS_PATH}")

def hierarchical_propagation(num_samples: int, num_classes: int, model_classes: np.ndarray):
    """Applies hierarchical max-propagation (Child score -> Parent score) to the raw predictions."""
    print("## Hierarchical Propagation")
    
    # Build GO DAG (Child -> Parent edges for upward propagation)
    graph = obonet.read_obo(Config.OBO_PATH)
    go_dag = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        if "is_a" in data:
            for parent in data["is_a"]:
                go_dag.add_edge(node, parent)

    # Map terms to column indices
    class_to_idx = {term: i for i, term in enumerate(model_classes)}

    # Determine propagation order (Root to Leaf)
    # The reversed topological sort ensures a term is processed only after all its ancestors 
    # (which might also be children of other terms) have had their scores potentially boosted.
    sorted_terms = [
        term for term in reversed(list(nx.topological_sort(go_dag)))
        if term in class_to_idx
    ]

    # Build propagation steps: (Child Index, Parent Index)
    prop_steps = []
    for child in sorted_terms:
        if child in go_dag:
            c_idx = class_to_idx[child]
            # Successors in our graph are the direct parents
            for parent in go_dag.successors(child):
                if parent in class_to_idx:
                    prop_steps.append((c_idx, class_to_idx[parent]))

    print(f"Propagation steps prepared: {len(prop_steps)}")
    
    # Propagate scores batch-wise
    fp_prop = np.memmap(
        Config.PROP_PREDS_PATH, 
        dtype="float32", 
        mode="w+", 
        shape=(num_samples, num_classes)
    )
    fp_raw_read = np.memmap(
        Config.RAW_PREDS_PATH, 
        dtype="float32", 
        mode="r", 
        shape=(num_samples, num_classes)
    )

    for i in tqdm(range(0, num_samples, Config.BATCH_SIZE), desc="Propagating scores"):
        end = min(i + Config.BATCH_SIZE, num_samples)

        # Load batch copy into RAM
        batch = np.array(fp_raw_read[i:end])

        # Apply propagation: P_parent = max(P_parent, P_child)
        for child_idx, parent_idx in prop_steps:
            batch[:, parent_idx] = np.maximum(
                batch[:, parent_idx], 
                batch[:, child_idx]
            )

        # Write propagated batch to output memmap
        fp_prop[i:end] = batch

        if i % (Config.BATCH_SIZE * 5) == 0:
            fp_prop.flush()
            time.sleep(0.05)

    fp_prop.flush()
    print(f"Propagated predictions saved to: {Config.PROP_PREDS_PATH}")


def write_submission_file(num_samples: int, test_ids: list, model_classes: np.ndarray, threshold: float):
    """Loads final propagated predictions, applies the threshold, and writes the TSV submission file."""
    print(f"## Writing Submission File (Threshold: {threshold:.4f})")
    
    fp_final = np.memmap(
        Config.PROP_PREDS_PATH, 
        dtype="float32", 
        mode="r", 
        shape=(num_samples, len(model_classes))
    )

    with open(Config.SUBMISSION_FILE, "w") as f:
        # Use a reasonable chunk size for writing
        chunk_size = 250 
        for i in tqdm(range(0, num_samples, chunk_size), desc="Writing submission"):
            end = min(i + chunk_size, num_samples)

            batch_probs = np.array(fp_final[i:end])
            batch_ids = test_ids[i:end]

            # Apply threshold and get coordinates (row, column) of positive predictions
            rows, cols = np.where(batch_probs >= threshold)

            buf = []
            # Iterate through positive predictions and format for TSV
            for r, c in zip(rows, cols):
                protein_id = batch_ids[r]
                go_term = model_classes[c]
                confidence_score = batch_probs[r, c]
                buf.append(f"{protein_id}\t{go_term}\t{confidence_score:.3f}\n")

            f.writelines(buf)

            # Flush periodically for safety
            if i % 2500 == 0:
                f.flush()
                os.fsync(f.fileno())
                time.sleep(0.05)

    print(f"Submission saved at: {Config.SUBMISSION_FILE}")


# --- NEW UTILITY FUNCTIONS ---
class ModelEMA:
    """ EMA logic ti improve model stability"""
    def __init__(self, model:nn.Module, decay:float=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

class ModelCheckpointer:
    """ Checkpoint logic to save the model"""
    def __init__(self, checkpoint_dir: Path, k: int=3):
        self.checkpoint_dir = checkpoint_dir
        self.k = k
        self.best_scores: list[tuple[float, Path]] = []
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def checkpoint(self, model: nn.Module, score: float, epoch: int):
        ckpt_path = self.checkpoint_dir / f"score_{score:.5f}_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        self.best_scores.append(score, ckpt_path)
        self.best_scores.sort(key=lambda x: -x[0]) # Sort descending

        while len(self.best_scores) > self.k:
            _, old_path = self.best_scores.pop()
            old_path.unlink(missing_ok=True)

def compute_fmax(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """ Compute the F1-macro score for the predictions"""
    thresholds = torch.arange(0.01, 1.0, 0.01, device=preds.device)
    best_f1 = 0.0

    for thr in thresholds:
        pred_binary = (preds >= thr).float()
        tp = (pred_binary * labels).sum()
        fp = (pred_binary * (1 - labels)).sum()
        fn = ((1 - pred_binary) * labels).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1 = max(best_f1, f1.item())

    return best_f1