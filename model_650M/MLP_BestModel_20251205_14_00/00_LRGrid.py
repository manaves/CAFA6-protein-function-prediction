import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter # tensorboard --logdir=runs
from tqdm import tqdm
from datetime import datetime
import copy # Added for deep copying the model state

# --- CONFIGURATION PARAMETERS ---
class Config:
    """A class to hold all configuration parameters."""
    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_embeddings_split.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_embeddings_split.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse.npy")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse.npy")
    
   
    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    WEIGHT_DECAY = 5e-3
    
    # --- SWEEP-SPECIFIC PARAMETERS ---
    NUM_EPOCHS = 200     # Original full training epochs (Used for Scheduler max)
    SWEEP_EPOCHS = 5    # Number of epochs to run for the quick sweep
    # The default LR will be overwritten in the sweep loop
    LEARNING_RATE = 1e-3 
    PATIENCE = 15        # Early stopping is disabled for the sweep
    # -----------------------------------
    
    HIDDEN_DIMS = [1024, 1024, 512] 
    DROPOUT_RATE = 0.4 
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True 
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "model_architecture_log.txt")
    
# Global paths (will be updated for each sweep run)
MODEL_SAVE_PATH = "" 
LOG_DIR = ""

print(f"Device: {Config.DEVICE}")

# --- UTILITY FUNCTIONS (Placeholder - Assume the functions from the previous prompt are here) ---
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
    if sp.issparse(Y_train):
        label_freq = np.asarray(Y_train.sum(axis=0)).ravel()
    else:
        label_freq = np.sum(Y_train, axis=0)

    label_freq = label_freq.astype(np.float32)
    label_freq[label_freq < 1.0] = 1.0
    
    pos_weight = (N - label_freq) / (label_freq + eps)
    return np.clip(pos_weight, a_min=1.0, a_max=100.0)

def load_and_preprocess_data():
    """Loads, normalizes data, and determines input/label dimensions."""
    print("Loading embeddings and labels...")
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

    print(f"Input dim: {input_dim}, labels: {label_count}")
    return X_train, X_val, Y_train_sparse, Y_val_sparse, input_dim, label_count

class SparseLabelDataset(Dataset):
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

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class ResidualMLP(nn.Module):
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
        encoded_x = self.encoder(x)
        logits = self.head(encoded_x)
        return logits

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, scaler, epoch: int, max_epochs: int):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False)

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
    
    # Thresholds for validation F1 tuning (Fixed set for quick comparison)
    thresholds = np.arange(0.30, 0.95, 0.05)
    T = len(thresholds)

    # Initialize per-threshold, per-label counters for metrics
    TP = np.zeros((T, label_count), dtype=np.int64)
    FP = np.zeros((T, label_count), dtype=np.int64)
    FN = np.zeros((T, label_count), dtype=np.int64)
    nan_detected = False

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)

            if Config.USE_AMP and Config.DEVICE.type == "cuda":
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
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
        # Return 0.0 scores if validation failed
        return 0.5, 0.0 

    # Compute macro F1 per threshold
    f1_per_threshold = np.zeros(T)
    for i in range(T):
        tp = TP[i].astype(float)
        fp = FP[i].astype(float)
        fn = FN[i].astype(float)
        
        denom = 2 * tp + fp + fn
        
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_label = np.where(denom > 0, (2 * tp) / denom, 0.0)
        
        f1_per_threshold[i] = np.mean(f1_label)

    # Select best threshold and F1
    best_idx = int(np.argmax(f1_per_threshold))
    best_t = float(thresholds[best_idx])
    best_f1_local = float(f1_per_threshold[best_idx])
    
    return best_t, best_f1_local

def save_checkpoint(*args, **kwargs):
    """Placeholder: Disabled for the fast sweep."""
    pass 


# --- CORE TRAINING FUNCTION FOR SWEEP ---

def run_training_experiment(lr: float, epoch_max: int, 
                            initial_model_state: dict, 
                            train_loader: DataLoader, val_loader: DataLoader, 
                            pos_weight_torch: torch.Tensor, 
                            LABEL_COUNT: int, INPUT_DIM: int) -> tuple:
    """
    Runs the full training process for a given Learning Rate (lr) for a fixed number of epochs.
    
    Returns: (final_val_f1, final_train_loss)
    """
    
    # 1. Setup paths and loggers for this specific LR run
    lr_str = f"{lr:.0e}".replace('-', 'n').replace('+', 'p') # e.g., 5e-05 -> 5en05
    log_dir = os.path.join("./runs", f"sweep_lr_{lr_str}_" + datetime.now().strftime("%H%M%S"))
    model_save_path = os.path.join(Config.OUTPUT_DIR, f"sweep_best_mlp_lr_{lr_str}.pth")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n--- Starting LR Sweep for: {lr:.5e} ---")
    
    # 2. Model Initialization (Reset model state for each LR run)
    full_dims = [INPUT_DIM] + Config.HIDDEN_DIMS
    model = ResidualMLP(
        full_dims, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)
    
    # Load initial state (important to ensure each run starts from the same point)
    model.load_state_dict(initial_model_state) 
    
    # 3. Criterion, Optimizer, Scheduler, Scaler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=Config.WEIGHT_DECAY
    )
    # T_max is set to the full NUM_EPOCHS for consistency in the schedule shape
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS) 
    scaler = torch.amp.GradScaler(enabled=(Config.USE_AMP and Config.DEVICE.type == 'cuda'))

    best_val_f1 = -1.0 # Reset best F1 for this run
    
    # 4. Training Loop (Fixed number of epochs for the sweep)
    final_loss = 0.0
    final_f1 = 0.0
    
    for epoch in range(1, epoch_max + 1):
        
        # Train
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, epoch_max)
        
        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Validate
        best_t_local, best_f1_local = validate_epoch(model, val_loader, LABEL_COUNT)

        print(
            f"  Ep {epoch}/{epoch_max} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {epoch_loss:.5f} | "
            f" Best threshold {best_t_local} | Val F1(macro): {best_f1_local:.5f}"
        )
        
        # Log to Tensorboard
        writer.add_scalar("Sweep/Train_Loss", epoch_loss, epoch)
        writer.add_scalar("Sweep/Val_F1_Macro", best_f1_local, epoch)
        writer.add_scalar("Sweep/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Step Scheduler (Only step if we are training more than 1 epoch)
        scheduler.step()
        
        # Save metrics from the final epoch
        if epoch == epoch_max:
            final_loss = epoch_loss
            final_f1 = best_f1_local
            
    writer.close()
    
    return final_f1, final_loss

# --- MAIN EXECUTION: LEARNING RATE SWEEP ---
if __name__ == "__main__":
    
    set_seed()
    
    # 1. Data Loading (Done only once)
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT = load_and_preprocess_data()
    N_train = len(X_train)

    # Calculate Pos Weight (Done only once)
    pos_weight_np = calculate_pos_weight(Y_train_sparse, N_train)
    pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)

    # Dataset and DataLoaders (Done only once)
    train_ds = SparseLabelDataset(X_train, Y_train_sparse)
    val_ds = SparseLabelDataset(X_val, Y_val_sparse)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Initialize Model (Used to reset state for each run)
    initial_model = ResidualMLP(
        [INPUT_DIM] + Config.HIDDEN_DIMS, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)
    
    # Save the initial random state to ensure all sweep runs start identically
    initial_model_state = copy.deepcopy(initial_model.state_dict())
    del initial_model # Free up VRAM

    # 3. Define the LR Sweep Grid
    LR_GRID = [
        1e-5,
        2e-5,
        5e-5,
        1e-4,
        2e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2
    ]
    
    print("\n" + "="*80)
    print(f"STARTING LEARNING RATE SWEEP ({Config.SWEEP_EPOCHS} EPOCHS PER RUN)")
    print(f"Testing LRs: {LR_GRID}")
    print("="*80)
    
    sweep_results = []
    
    for lr in LR_GRID:
        try:
            # Run the training experiment
            final_f1, final_loss = run_training_experiment(
                lr=lr,
                epoch_max=Config.SWEEP_EPOCHS,
                initial_model_state=initial_model_state,
                train_loader=train_loader,
                val_loader=val_loader,
                pos_weight_torch=pos_weight_torch,
                LABEL_COUNT=LABEL_COUNT,
                INPUT_DIM=INPUT_DIM
            )
            
            sweep_results.append({
                'LR': lr, 
                f'F1_Ep{Config.SWEEP_EPOCHS}': final_f1, 
                f'Loss_Ep{Config.SWEEP_EPOCHS}': final_loss
            })

        except RuntimeError as e:
            print(f"--- RUNTIME ERROR for LR {lr:.5e}: {e} ---")
            print("Skipping this LR...")
            sweep_results.append({'LR': lr, 'F1': 0.0, 'Loss': float('nan'), 'Error': str(e)})


    # 4. Display Final Sweep Results
    print("\n" + "="*80)
    print("FINAL SWEEP RESULTS")
    print("="*80)
    
    best_lr = None
    best_f1_overall = -1.0
    
    print(f"{'LR':<10} | {'Final F1 (Macro)':<20} | {'Final Loss':<15}")
    print("-" * 50)
    for result in sweep_results:
        f1_key = f'F1_Ep{Config.SWEEP_EPOCHS}'
        loss_key = f'Loss_Ep{Config.SWEEP_EPOCHS}'
        
        f1 = result.get(f1_key, 0.0)
        loss = result.get(loss_key, float('nan'))
        
        print(f"{result['LR']:<10.1e} | {f1:<20.5f} | {loss:<15.5f}")
        
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_lr = result['LR']

    print("-" * 50)
    if best_lr:
        print(f"RECOMMENDED LR: {best_lr:.1e} (Highest F1 after {Config.SWEEP_EPOCHS} epochs)")
    else:
        print("No successful runs found.")
    print("="*80)