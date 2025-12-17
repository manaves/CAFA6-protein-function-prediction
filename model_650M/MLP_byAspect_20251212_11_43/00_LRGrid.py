import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import copy # Added for deep copying the model state

from config import Config
from utils import (
    set_seed, apply_label_smoothing,
    calculate_pos_weight, load_and_preprocess_data, load_go_dag, map_model_classes,
    get_propagation_steps, SparseLabelDataset, ResidualMLP, validate_cafa_pk
)
    
# Global paths (will be updated for each sweep run)
MODEL_SAVE_PATH = "" 
LOG_DIR = ""

print(f"Device: {Config.DEVICE}")

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
            
        # Validate (Using CAFA-PK validation function)
        # We use step=0.05 for speed during training
        best_t_local, best_f1_local = validate_cafa_pk(
            model, val_loader, ia_weights_np=ia_weights_np, 
            propagation_steps=propagation_steps, 
            label_count=LABEL_COUNT, device=Config.DEVICE,
            th_step=0.05  # <--- Optimization of speed
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
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT, ALL_GO_TERMS_LIST = load_and_preprocess_data()
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
        1e-4, 2e-4, 5e-4, 
        1e-3, 2e-3, 5e-3
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