import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from config import Config
from utils import (
    set_seed, normalize_rows, safe_load_sparse_array, apply_label_smoothing,
    calculate_pos_weight, load_and_preprocess_data, load_go_dag, map_model_classes,
    get_propagation_steps, SparseLabelDataset, ResidualMLP, validate_cafa_pk_gpu,
    validate_epoch_propagated
)

print(f"Device: {Config.DEVICE}")

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
    the resulting scores to a memory-mapped file and a TSV file with protein IDs.
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

    # Load protein IDs from validation split CSV
    val_split_path = os.path.join(Config.INPUT_VAL_DIR, "val_fold0_split.csv")
    logging.info(f"Loading validation protein IDs from: {val_split_path}")
    try:
        val_df = pd.read_csv(val_split_path)
        protein_ids = val_df['id'].values.tolist()
        if len(protein_ids) != total_samples:
            logging.warning(
                f"Number of protein IDs ({len(protein_ids)}) doesn't match "
                f"number of samples ({total_samples}). Using min of both."
            )
            min_len = min(len(protein_ids), total_samples)
            protein_ids = protein_ids[:min_len]
            total_samples = min_len
        logging.info(f"Loaded {len(protein_ids)} protein IDs")
    except FileNotFoundError:
        logging.warning(f"Validation split CSV not found at {val_split_path}. Cannot save TSV with protein IDs.")
        protein_ids = None
    except Exception as e:
        logging.warning(f"Error loading protein IDs: {e}. Continuing without protein IDs in TSV.")
        protein_ids = None

    # Load model classes (GO terms)
    model_classes = np.load(Config.CLASSES_PATH)

    # Create memory-mapped file
    logging.info(f"Creating memmap for propagated predictions at: {Config.PREDICTIONS_MEMMAP_PATH} with shape ({total_samples}, {label_count})")
    fp = np.memmap(
        Config.PREDICTIONS_MEMMAP_PATH, 
        dtype='float32', 
        mode='w+', 
        shape=(total_samples, label_count)
    )

    # Prepare list for TSV output
    tsv_predictions = []

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
            
            # 4. Save the propagated scores to memmap
            batch_size = scores.shape[0]
            fp[start_idx : start_idx + batch_size, :] = scores
            
            # 5. Collect predictions for TSV file (with protein IDs)
            if protein_ids is not None:
                for i in range(batch_size):
                    protein_id = protein_ids[start_idx + i]
                    # Save all predictions above a minimum threshold
                    min_threshold = 0.01
                    above_threshold = scores[i] >= min_threshold
                    predicted_indices = np.where(above_threshold)[0]
                    
                    for pred_idx in predicted_indices:
                        go_term = model_classes[pred_idx]
                        prediction_score = float(scores[i, pred_idx])
                        tsv_predictions.append({
                            'protein_id': protein_id,
                            'go_term': go_term,
                            'prediction': prediction_score
                        })
            
            start_idx += batch_size

    fp.flush() # Ensure data is written to disk
    logging.info(f"Propagated scores saved in: {Config.PREDICTIONS_MEMMAP_PATH}")
    logging.info(f"Best F1 (val): {best_val_f1:.5f} with threshold {best_threshold}")
    
    # Save TSV file with protein IDs
    if protein_ids is not None and len(tsv_predictions) > 0:
        tsv_output_path = os.path.join(Config.OUTPUT_DIR, "val_predictions_with_ids.tsv")
        logging.info(f"Saving predictions with protein IDs to: {tsv_output_path}")
        
        df_predictions = pd.DataFrame(tsv_predictions)
        # Sort by protein_id and then by prediction score (descending)
        df_predictions = df_predictions.sort_values(
            by=['protein_id', 'prediction'],
            ascending=[True, False]
        )
        
        df_predictions.to_csv(
            tsv_output_path,
            sep='\t',
            index=False,
            header=True
        )
        
        logging.info(f"Saved {len(df_predictions)} predictions for {len(protein_ids)} proteins to TSV")
    elif protein_ids is None:
        logging.info("Skipping TSV file creation (protein IDs not available)")

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

        # --- OFFICIAL VALIDATION (CAFA-PK) ---
        # We use step=0.05 for speed during training
        best_t_local, best_f1_local = validate_cafa_pk_gpu(
            model, val_loader, ia_weights_np=ia_weights_np, 
            propagation_steps=PROPAGATION_STEPS, 
            label_count=LABEL_COUNT, device=Config.DEVICE
        )

        logging.info(
            f"Epoch {epoch} | Train Loss: {epoch_loss:.5f} | "
            f"Fmax: {best_f1_local:.5f} (Th: {best_t_local:.2f})"
        )
        
        # Metrics to tensorboard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Metrics/Fmax_ProteinCentric", best_f1_local, epoch)
        writer.add_scalar("Metrics/Best_Threshold", best_t_local, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Step Scheduler
        scheduler.step()

        # Checkpoint/Early Stopping Logic (Guided by the REAL metric)
        if best_f1_local > best_val_f1 + 1e-4: # Small delta to avoid noise
            best_val_f1 = best_f1_local
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_t_local, best_val_f1)
        else:
            epochs_no_improve += 1
            logging.info(f"  No improvement ({epochs_no_improve}/{Config.PATIENCE})")

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