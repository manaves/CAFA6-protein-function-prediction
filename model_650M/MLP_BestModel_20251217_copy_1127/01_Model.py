import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import time

# --- CONFIGURATION PARAMETERS ---
class Config:
    """A class to hold all configuration parameters."""
    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_embeddings_new_split.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_embeddings_new_split.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse_new_split.npz")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse_new_split.npz")
    
    # Log path
    LOG_DIR = os.path.join("./runs", "mlp_best_model_" + time.strftime("%Y%m%d_%H_%M")) #tensorboard --logdir=runs --reload_interval 5
    LOG_FILE = "train_log.txt"

    # Output Paths
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_predictions.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 200
    PATIENCE = 15  # Early stopping patience
    EMBEDDING_DIM = 1280
    HIDDEN_DIMS = [1024, 1024, 512]
    DROPOUT_RATE = 0.4 # Best score with 0.4
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True  # Automatic Mixed Precision (AMP)
    
print(f"Device: {Config.DEVICE}")

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
    obj = sp.load_npz(path)
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
        for i in range(1, len(dims)-1):
            dim_in = dims[i]
            dim_out = dims[i+1]
            
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits: head(encoder(x))"""
        encoded_x = self.encoder(x)
        logits = self.head(encoded_x)
        return logits

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
    
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    set_seed()
    
    # Data Loading and Prep
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT = load_and_preprocess_data()
    N_train = len(X_train)

    # Calculate Pos Weight
    pos_weight_np = calculate_pos_weight(Y_train_sparse, N_train)
    pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)

    # Dataset and DataLoaders
    train_ds = SparseLabelDataset(X_train, Y_train_sparse)
    val_ds = SparseLabelDataset(X_val, Y_val_sparse)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    # Model, Criterion, Optimizer, Scheduler, Scaler
    # Use INPUT_DIM as the first dimension in HIDDEN_DIMS for initialization
    # Note: original code uses HIDDEN_DIMS directly which is correct 
    # since ResidualMLP takes care of the first linear layer.
    full_dims = [INPUT_DIM] + Config.HIDDEN_DIMS 
    model = ResidualMLP(
        full_dims, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=(Config.USE_AMP and Config.DEVICE.type == 'cuda'))

        
    # Tensorboard writer initialization
    print(f"Initializing TensorBoard writer at: {Config.LOG_DIR}")
    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    
    writer.add_hparams(
        {'lr': Config.LEARNING_RATE, 'batch_size': Config.BATCH_SIZE, 'dropout': Config.DROPOUT_RATE}, 
        {'init/f1_score': 0.0}
    )

    # Training Loop
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        
        # Train
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        
        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Validate
        best_t_local, best_f1_local = validate_epoch(model, val_loader, LABEL_COUNT)

        print(
            f"Epoch {epoch} | Train Loss: {epoch_loss:.5f} | "
            f"Val Best Th: {best_t_local:.3f} | Val F1(macro): {best_f1_local:.5f}"
        )
        
        # Metrics to tensorboard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Metrics/Val_F1_Macro", best_f1_local, epoch)
        writer.add_scalar("Metrics/Best_Threshold", best_t_local, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Step Scheduler
        scheduler.step()

        # Checkpoint/Early Stopping Logic
        if best_f1_local > best_val_f1 + 1e-5:
            best_val_f1 = best_f1_local
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_t_local, best_val_f1)
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{Config.PATIENCE})")

        if epochs_no_improve >= Config.PATIENCE:
            print("Early stopping activated.")
            break
        
        writer.close()
            
    # Final Prediction Saving
    save_predictions_to_memmap(model, val_loader, len(val_ds), LABEL_COUNT)