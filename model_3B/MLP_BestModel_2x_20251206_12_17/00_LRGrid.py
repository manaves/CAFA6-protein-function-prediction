import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
import networkx as nx
import obonet
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    IA_WEIGHTS_PATH = os.path.join(INPUT_DIR, "ia_weights.npy")
    OBO_FILE_PATH = "../../general_input/train/go-basic.obo"
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes.npy")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    WEIGHT_DECAY = 5e-3
    
    # --- SWEEP-SPECIFIC PARAMETERS ---
    NUM_EPOCHS = 200     # Original full training epochs (Used for Scheduler max)
    SWEEP_EPOCHS = 5     # Number of epochs to run for the quick sweep
    # The default LR will be overwritten in the sweep loop
    LEARNING_RATE = 1e-3 
    PATIENCE = 15        # Early stopping is disabled for the sweep
    # -----------------------------------
    
    HIDDEN_DIMS = [2048, 2048, 1024] 
    DROPOUT_RATE = 0.4 
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True 
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "LR_sweep_log.txt")
    
# Global paths (will be updated for each sweep run)
MODEL_SAVE_PATH = "" 
LOG_DIR = ""

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

# --- PROPAGATION CODE ---
def load_go_dag(obo_path: str) -> nx.DiGraph:
    """
    Reads the Gene Ontology OBO file and converts it to a NetworkX DiGraph.
    Edges are defined from **child to parent** (for upward propagation).
    """
    print("1. Reading GO file and building DAG...")
    graph = obonet.read_obo(obo_path)
    
    # Initialize a new DiGraph for propagation: Child -> Parent
    go_dag = nx.DiGraph()
    
    for node, data in graph.nodes(data=True):
        go_dag.add_node(node)
        
        # 'is_a' relationships define parent terms
        if 'is_a' in data:
            for parent in data['is_a']:
                # Add edge: Child -> Parent
                go_dag.add_edge(node, parent)
                
    print(f"Graph loaded: {go_dag.number_of_nodes()} nodes.")
    return go_dag

def map_model_classes(classes_path: str, go_dag: nx.DiGraph) -> tuple[dict, list]:
    """
    Loads model class IDs and maps GO terms to their column indices.
    """
    print("2. Mapping classes from the model to GO terms...")
    model_classes = np.load(classes_path)
    
    # Mapping GO Term ID -> Column Index
    class_to_idx = {term: i for i, term in enumerate(model_classes)}

    # Identify which terms from the model are actually in the GO DAG
    valid_terms = [term for term in model_classes if term in go_dag]
    
    print(f"Valid terms found in DAG: {len(valid_terms)} out of {len(model_classes)}")
    return class_to_idx, model_classes

def get_propagation_steps(go_dag: nx.DiGraph, class_to_idx: dict) -> list[tuple[int, int]]:
    """
    Determines the ordered steps (child_idx, parent_idx) needed for score propagation.
    The order ensures child scores are processed before their parents.
    """
    print("3. Establishing topological order for propagation...")
    
    # Topological sort (Child -> Parent): ensures that children are processed before parents
    full_topological_order = list(nx.topological_sort(go_dag))
    
    # Iterate from root-most terms down to leaf terms by reversing the order
    sorted_model_terms = [term for term in reversed(full_topological_order) if term in class_to_idx]

    propagation_steps = []
    
    print("Topological order established. Generating propagation steps...")
    
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

    print(f"Parent-child relationships to be processed: {len(propagation_steps)}")
    return propagation_steps

# --- DATASET AND MODEL CLASSES ---
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

# --- TRAINING AND VALIDATION FUNCTIONS ---
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

def validate_epoch_propagated(model: nn.Module, loader: DataLoader, label_count: int, 
                              ia_weights_np: np.ndarray, 
                              propagation_steps: list[tuple[int, int]]) -> tuple:
    """
    Performs validation, applies GO propagation to predictions, and computes the 
    best WEIGHTED F1-score.
    """
    model.eval()
    
    # Thresholds for validation F1 tuning
    thresholds = np.arange(0.35, 0.95, 0.05)
    T = len(thresholds)

    # Initialize per-threshold, per-label counters for True Positives, False Positives, False Negatives
    TP = np.zeros((T, label_count), dtype=np.float64)
    FP = np.zeros((T, label_count), dtype=np.float64)
    FN = np.zeros((T, label_count), dtype=np.float64)
    
    nan_detected = False

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Validation"):
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)

            # 1. Obtener Logits
            if Config.USE_AMP and Config.DEVICE.type == "cuda":
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_detected = True
                break

            # 2. Convertir a Probabilidades (scores) y Numpy
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = yb.cpu().numpy().astype(np.int8) # Targets binarios

            # 3. Aplicar Propagación en el batch (usando numpy)
            # parent_score = max(parent_score, child_score)
            for child_idx, parent_idx in propagation_steps:
                probs[:, parent_idx] = np.maximum(
                    probs[:, parent_idx], 
                    probs[:, child_idx]
                )

            # 4. Calcular Métricas para cada umbral (sobre las probabilidades PROPAGADAS)
            for i, t in enumerate(thresholds):
                # Clasificación: preds = 1 si prob >= t
                preds = (probs >= t).astype(np.int8)
                
                # Calcular y acumular contadores ponderados (weighted counts)
                TP_batch = (preds == 1) & (targets == 1)
                FP_batch = (preds == 1) & (targets == 0)
                FN_batch = (preds == 0) & (targets == 1)
                
                # Acumular los recuentos ponderados (multiplicar por IA weights)
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
        
        # Weighted Precision (P_w)
        P_w = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Weighted Recall (R_w)
        R_w = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Weighted F1
        F1_w = (2 * P_w * R_w) / (P_w + R_w) if (P_w + R_w) > 0 else 0.0
        
        weighted_f1_per_threshold[i] = F1_w

    # Select best threshold and F1
    best_idx = int(np.argmax(weighted_f1_per_threshold))
    best_t = float(thresholds[best_idx])
    best_f1_local = float(weighted_f1_per_threshold[best_idx])
    
    return best_t, best_f1_local

def save_checkpoint(*args, **kwargs):
    """Placeholder: Disabled for the fast sweep."""
    pass 


# --- CORE TRAINING FUNCTION FOR SWEEP ---

def run_training_experiment(lr: float, epoch_max: int, 
                            initial_model_state: dict, 
                            train_loader: DataLoader, val_loader: DataLoader, 
                            pos_weight_torch: torch.Tensor, 
                            LABEL_COUNT: int, INPUT_DIM: int,
                            propagation_steps: list[tuple[int, int]]) -> tuple:
    """
    Runs the full training process for a given Learning Rate (lr) for a fixed number of epochs.
    
    Returns: (final_val_f1, final_train_loss)
    """
    
    # 1. Setup paths and loggers for this specific LR run
    lr_str = f"{lr:.0e}".replace('-', 'n').replace('+', 'p') # e.g., 5e-05 -> 5en05
    log_dir = os.path.join("./runs", f"sweep_lr_{lr_str}_" + datetime.now().strftime("%H%M%S"))
    
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
    
    ia_weights_np = np.load(Config.IA_WEIGHTS_PATH)

    for epoch in range(1, epoch_max + 1):
        
        # Train
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, epoch_max)
        
        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            
        # Validate (Using the propagated validation function)
        best_t_local, best_f1_local = validate_epoch_propagated(
            model, 
            val_loader, 
            LABEL_COUNT, 
            ia_weights_np,
            propagation_steps
        )

        print(
            f"  Ep {epoch}/{epoch_max} | LR: {optimizer.param_groups[0]['lr']:.2e} | Train Loss: {epoch_loss:.5f} | "
            f" Best threshold {best_t_local} | Val F1(W): {best_f1_local:.5f}"
        )
        
        # Log to Tensorboard
        writer.add_scalar("Sweep/Train_Loss", epoch_loss, epoch)
        writer.add_scalar("Sweep/Val_F1_Weighted", best_f1_local, epoch)
        writer.add_scalar("Sweep/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Step Scheduler
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

    # --- 2. CONFIGURACIÓN DE LA PROPAGACIÓN ---
    print("\n--- Initializing GO Propagation Configuration ---")
    go_dag = load_go_dag(Config.OBO_FILE_PATH)
    class_to_idx, _ = map_model_classes(Config.CLASSES_PATH, go_dag)
    propagation_steps = get_propagation_steps(go_dag, class_to_idx)
    del go_dag, class_to_idx # Free memory
    print("--- GO Propagation Configuration Complete ---")

    # 3. Initialize Model (Used to reset state for each run)
    initial_model = ResidualMLP(
        [INPUT_DIM] + Config.HIDDEN_DIMS, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)
    
    # Save the initial random state to ensure all sweep runs start identically
    initial_model_state = copy.deepcopy(initial_model.state_dict())
    del initial_model # Free up VRAM

    # 4. Define the LR Sweep Grid
    LR_GRID = [
        1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 
        1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2
    ]
    
    print("\n" + "="*80)
    print(f"STARTING LEARNING RATE SWEEP ({Config.SWEEP_EPOCHS} EPOCHS PER RUN) with GO PROPAGATION")
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
                INPUT_DIM=INPUT_DIM,
                propagation_steps=propagation_steps
            )
            
            sweep_results.append({
                'LR': lr, 
                f'F1_Ep{Config.SWEEP_EPOCHS}': final_f1, 
                f'Loss_Ep{Config.SWEEP_EPOCHS}': final_loss
            })

        except RuntimeError as e:
            print(f"--- RUNTIME ERROR for LR {lr:.5e}: {e} ---")
            print("Skipping this LR...")
            sweep_results.append({'LR': lr, f'F1_Ep{Config.SWEEP_EPOCHS}': 0.0, f'Loss_Ep{Config.SWEEP_EPOCHS}': float('nan'), 'Error': str(e)})


    # 5. Display Final Sweep Results
    print("\n" + "="*80)
    print("FINAL SWEEP RESULTS")
    print("="*80)
    
    best_lr = None
    best_f1_overall = -1.0
    
    f1_key = f'F1_Ep{Config.SWEEP_EPOCHS}'
    loss_key = f'Loss_Ep{Config.SWEEP_EPOCHS}'

    print(f"{'LR':<10} | {'Final F1 (Weighted)':<20} | {'Final Loss':<15}")
    print("-" * 50)
    for result in sweep_results:
        f1 = result.get(f1_key, 0.0)
        loss = result.get(loss_key, float('nan'))
        
        f1_display = f"{f1:.5f}" if not np.isnan(f1) else "N/A (Error)"
        loss_display = f"{loss:.5f}" if not np.isnan(loss) else "N/A (Error)"
        
        print(f"{result['LR']:<10.1e} | {f1_display:<20} | {loss_display:<15}")
        
        if not np.isnan(f1) and f1 > best_f1_overall:
            best_f1_overall = f1
            best_lr = result['LR']

    print("-" * 50)
    if best_lr:
        print(f"RECOMMENDED LR: {best_lr:.1e} (Highest F1 after {Config.SWEEP_EPOCHS} epochs)")
    else:
        print("No successful runs found.")
    print("="*80)