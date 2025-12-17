import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score
import logging
import sys
import networkx as nx # Added for GO DAG
import obonet # Added for GO DAG

# --- CONFIGURATION PARAMETERS ---
class Config:
    """A class to hold all configuration parameters."""
    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")
    
    # GO Hierarchy Paths (NOTE: You might need to adjust the OBO path)
    # Suponiendo que general_input está al mismo nivel que tu directorio principal
    OBO_FILE_PATH = "../../general_input/train/go-basic.obo" 
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes.npy")

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_embeddings_split.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_embeddings_split.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse.npy")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse.npy")
    IA_WEIGHTS_PATH = os.path.join(INPUT_DIR, "ia_weights.npy")
    
    # Log path
    LOG_DIR = os.path.join("./runs", "mlp_best_model_3B_" + time.strftime("%Y%m%d_%H_%M"))
    LOG_FILE = "train_log.txt"

    # Output Paths
    # IMPORTANT: Change PREDICTIONS_MEMMAP_PATH to clearly indicate PROPGATED predictions
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_predictions_propagated.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_3B_2x.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4 # Best LR from previous experiments
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 400
    PATIENCE = 75 
    HIDDEN_DIMS = [2048, 2048, 1024]
    DROPOUT_RATE = 0.4 
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True 
    
print(f"Device: {Config.DEVICE}")

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

def validate_epoch_propagated(model: nn.Module, loader: DataLoader, label_count: int, 
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
    logging.info("  -> New best model saved.")

def save_propagated_predictions(model, loader: DataLoader, total_samples: int, label_count: int, propagation_steps: list[tuple[int, int]]):
    """
    Loads best model, applies GO propagation to the predictions, and saves 
    the resulting scores to a memory-mapped file.
    """
    
    # Load best model checkpoint
    logging.info("Loading best model for final prediction saving...")
    try:
        ckpt = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        best_threshold = ckpt.get('threshold', 0.5)
        best_val_f1 = ckpt.get('val_f1', 0.0)
    except FileNotFoundError:
        logging.error("Model checkpoint not found. Cannot save predictions.")
        return

    # Create memory-mapped file
    logging.info(f"Creating memmap for propagated predictions at: {Config.PREDICTIONS_MEMMAP_PATH} with shape ({total_samples}, {label_count})")
    fp = np.memmap(
        Config.PREDICTIONS_MEMMAP_PATH, 
        dtype='float32', 
        mode='w+', 
        shape=(total_samples, label_count)
    )

    model.eval()
    start_idx = 0
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Saving PROPGATED predictions..."):
            xb = xb.to(Config.DEVICE)
            
            # 1. Get Logits/Scores
            if Config.USE_AMP and Config.DEVICE.type == 'cuda':
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)
            
            # 2. Convert logits to probabilities (scores)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            # 3. APPLY GO PROPAGATION (Maximum Rule)
            for child_idx, parent_idx in propagation_steps:
                scores[:, parent_idx] = np.maximum(
                    scores[:, parent_idx], 
                    scores[:, child_idx]
                )
            
            # 4. Save the propagated scores
            batch_size = scores.shape[0]
            fp[start_idx : start_idx + batch_size, :] = scores
            start_idx += batch_size

    fp.flush() # Ensure data is written to disk
    logging.info(f"Propagated scores saved in: {Config.PREDICTIONS_MEMMAP_PATH}")
    logging.info(f"Best F1 (val): {best_val_f1:.5f} with threshold {best_threshold}")

# --- LOGGING SETUP ---
class TqdmLoggingHandler(logging.Handler):
    """Custom handler to allow logging messages to display above the tqdm progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logging(log_dir: str, log_file: str):
    """Initializes the Python logging system."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) 
    
    # Remove all existing handlers to prevent duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Full log saved to: {log_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 0. Setup and Data Prep
    setup_logging(Config.LOG_DIR, Config.LOG_FILE)
    logging.info(f"Device: {Config.DEVICE}")
    set_seed()
    
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT = load_and_preprocess_data()
    N_train = len(X_train)
    logging.info(f"Input dim: {INPUT_DIM}, labels: {LABEL_COUNT}")
    
    pos_weight_np = calculate_pos_weight(Y_train_sparse, N_train)
    pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)

    train_ds = SparseLabelDataset(X_train, Y_train_sparse)
    val_ds = SparseLabelDataset(X_val, Y_val_sparse)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 1. GO Propagation Initialization
    logging.info("\n--- Initializing GO Propagation Configuration ---")
    go_dag = load_go_dag(Config.OBO_FILE_PATH)
    class_to_idx, _ = map_model_classes(Config.CLASSES_PATH, go_dag)
    # The propagation_steps are required for both validation and final prediction saving
    PROPAGATION_STEPS = get_propagation_steps(go_dag, class_to_idx)
    del go_dag, class_to_idx # Free memory
    logging.info("--- GO Propagation Configuration Complete ---")

    # 2. Model, Criterion, Optimizer, Scheduler, Scaler
    full_dims = [INPUT_DIM] + Config.HIDDEN_DIMS 
    model = ResidualMLP(
        full_dims, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=(Config.USE_AMP and Config.DEVICE.type == 'cuda'))
        
    writer = SummaryWriter(log_dir=Config.LOG_DIR)
    writer.add_hparams(
        {'lr': Config.LEARNING_RATE, 'batch_size': Config.BATCH_SIZE, 'dropout': Config.DROPOUT_RATE}, 
        {'init/f1_score': 0.0}
    )

    # 3. Training Loop
    best_val_f1 = 0.0
    epochs_no_improve = 0
    ia_weights_np = np.load(Config.IA_WEIGHTS_PATH)
    
    logging.info("--- Starting Training Loop ---")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        
        # Train
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        
        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Validate with PROPAGATION
        best_t_local, best_f1_local = validate_epoch_propagated(
            model, 
            val_loader, 
            LABEL_COUNT, 
            ia_weights_np=ia_weights_np,
            propagation_steps=PROPAGATION_STEPS
        )

        logging.info(
            f"Epoch {epoch} | Train Loss: {epoch_loss:.5f} | "
            f"Val Best Th: {best_t_local:.3f} | Val F1(W) Propagated: {best_f1_local:.5f}"
        )
        
        # Metrics to tensorboard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Metrics/Val_F1_Weighted_Propagated", best_f1_local, epoch)
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
            logging.info(f"  No improvement ({epochs_no_improve}/{Config.PATIENCE})")

        if epochs_no_improve >= Config.PATIENCE:
            logging.info("Early stopping activated.")
            break
            
    writer.close()
    
    logging.info("Training finished. Final predictions saving (Propagated)...")
            
    # 4. Final Prediction Saving (Propagated Scores)
    save_propagated_predictions(
        model, 
        val_loader, 
        len(val_ds), 
        LABEL_COUNT,
        propagation_steps=PROPAGATION_STEPS
    )