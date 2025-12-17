import os
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from utils import *

# --- GO TERMS AND ASPECTS MAPPING ---
def get_aspect_labels(all_go_terms: list[str]) -> dict[str, list[str]]:
    """
    Load the train_terms.tsv file and map the GO terms to their corresponding aspects.
    """
    df_terms = pd.read_csv(Config.TRAIN_TERMS_PATH, sep='\t', names=['id', 'go_terms', 'aspect'], skiprows=1)
    term_aspect_map = df_terms[['go_terms', 'aspect']].drop_duplicates().set_index('go_terms')['aspect'].to_dict()
    
    bp_labels, mf_labels, cc_labels = [], [], []
    
    for term in all_go_terms:
        aspect = term_aspect_map.get(term)
        if aspect == 'P':
            bp_labels.append(term)
        elif aspect == 'F':
            mf_labels.append(term)
        elif aspect == 'C':
            cc_labels.append(term)
            
    print(f"Total terms: BP={len(bp_labels)}, MF={len(mf_labels)}, CC={len(cc_labels)}")
    
    return {
        'BP': bp_labels,
        'MF': mf_labels,
        'CC': cc_labels
    }

def get_aspect_propagation_steps(go_dag: nx.DiGraph, aspect_labels: list[str]):
    """
    Create the propagation steps (child_idx, parent_idx) using the specific indices of the aspect labels.
    """
    
    aspect_to_idx = {term: i for i, term in enumerate(aspect_labels)}
    propagation_steps = []
    
    for term in aspect_labels:
        child_idx = aspect_to_idx[term]
        
        for parent_term in go_dag.successors(term):
            if parent_term in aspect_to_idx:
                parent_idx = aspect_to_idx[parent_term]
                propagation_steps.append((child_idx, parent_idx))
    
    return sorted(list(set(propagation_steps)))

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

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, threshold: float, val_f1: float, filename: str):
    """Saves the model state, optimizer state, and metrics."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "threshold": threshold,
        "val_f1": val_f1,
    }
    torch.save(checkpoint, filename)
    logging.info("  -> New best model saved.")

def save_propagated_predictions(model, loader: DataLoader, total_samples: int, label_count: int, propagation_steps: list[tuple[int, int]], filename: str, checkpoint_path: str):
    """
    Loads best model, applies GO propagation to the predictions, and saves 
    the resulting scores to a memory-mapped file.
    """
    
    # Load best model checkpoint
    logging.info("Loading best model for final prediction saving...")
    try:
        ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        best_threshold = ckpt.get('threshold', 0.5)
        best_val_f1 = ckpt.get('val_f1', 0.0)
    except FileNotFoundError:
        logging.error(f"Model checkpoint not found at {checkpoint_path}. Cannot save predictions.")
        return

    # Create memory-mapped file using 'filename'
    logging.info(f"Creating memmap at: {filename} with shape ({total_samples}, {label_count})")
    fp = np.memmap(
        filename, 
        dtype='float32', 
        mode='w+', 
        shape=(total_samples, label_count)
    )

    model.eval()
    start_idx = 0
    with torch.no_grad():
        for xb, _ in tqdm(loader, desc="Saving PROPAGATED predictions..."):
            xb = xb.to(Config.DEVICE)
            
            if Config.USE_AMP and Config.DEVICE.type == 'cuda':
                with torch.autocast(device_type=Config.DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)
            
            scores = torch.sigmoid(logits).cpu().numpy()
            
            # Propagation
            for child_idx, parent_idx in propagation_steps:
                scores[:, parent_idx] = np.maximum(
                    scores[:, parent_idx], 
                    scores[:, child_idx]
                )
            
            batch_size = scores.shape[0]
            fp[start_idx : start_idx + batch_size, :] = scores
            start_idx += batch_size

    fp.flush()
    logging.info(f"Propagated scores saved in: {filename}")
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

if __name__ == "__main__":    
    setup_logging(Config.LOG_DIR, Config.LOG_FILE)
    logging.info(f"Device: {Config.DEVICE}")
    set_seed()

    # 1. LOAD FULL DATA
    try:
        X_train, X_val, Y_train_sparse_full, Y_val_sparse_full, INPUT_DIM, _, ALL_GO_TERMS_LIST = load_and_preprocess_data()
    except Exception as e:
        logging.error(f"Data load error: {e}")
        raise e

    logging.info(f"Input Dim: {INPUT_DIM}, Total Terms: {len(ALL_GO_TERMS_LIST)}")

    # 2. INIT GLOBAL RESOURCES
    go_dag = load_go_dag(Config.OBO_FILE_PATH)
    ASPECT_LABELS_MAP = get_aspect_labels(ALL_GO_TERMS_LIST)
    ia_weights_np_full = np.load(Config.IA_WEIGHTS_PATH)

    ASPECTS = ['BP', 'MF', 'CC']

    # 3. ASPECT LOOP
    
    for aspect in ASPECTS:
        logging.info(f"\n\n{'='*20} STARTING ASPECT: {aspect} {'='*20}")
        
        # --- A. FILTER DATA ---
        target_labels = ASPECT_LABELS_MAP[aspect]
        
        # Get column indices for this aspect
        indices_to_keep = [ALL_GO_TERMS_LIST.index(t) for t in target_labels if t in ALL_GO_TERMS_LIST]
        indices_to_keep = sorted(indices_to_keep)
        
        if not indices_to_keep:
            logging.warning(f"No labels found for aspect {aspect}. Skipping.")
            continue

        # Slice the sparse matrices (Columns)
        Y_train_aspect = Y_train_sparse_full[:, indices_to_keep]
        Y_val_aspect = Y_val_sparse_full[:, indices_to_keep]
        ia_weights_aspect = ia_weights_np_full[indices_to_keep]

        # --- B. REMOVE ZERO-POSITIVE LABELS (STABILITY FIX) ---
        if sp.issparse(Y_train_aspect):
            col_sums = np.array(Y_train_aspect.sum(axis=0)).flatten()
        else:
            col_sums = Y_train_aspect.sum(axis=0)
            
        valid_cols = col_sums > 0
        n_removed = len(valid_cols) - valid_cols.sum()
        
        if n_removed > 0:
            logging.info(f"Removing {n_removed} labels with 0 positives in train set.")
            Y_train_aspect = Y_train_aspect[:, valid_cols]
            Y_val_aspect = Y_val_aspect[:, valid_cols]
            ia_weights_aspect = ia_weights_aspect[valid_cols]
            
            # Update indices map to keep track of remaining labels
            indices_to_keep = np.array(indices_to_keep)[valid_cols].tolist()
            # Update local label list for propagation mapping
            target_labels = [target_labels[i] for i, v in enumerate(valid_cols) if v]

        LABEL_COUNT_ASPECT = Y_train_aspect.shape[1]
        logging.info(f"Final Label Count for {aspect}: {LABEL_COUNT_ASPECT}")

        # --- C. SETUP TRAINING ---
        pos_weight_np = calculate_pos_weight(Y_train_aspect, Y_train_aspect.shape[0])
        pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)

        train_ds = SparseLabelDataset(X_train, Y_train_aspect)
        val_ds = SparseLabelDataset(X_val, Y_val_aspect)
        
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = ResidualMLP(
            [INPUT_DIM] + Config.HIDDEN_DIMS,
            dropout=Config.DROPOUT_RATE,
            output_dim=LABEL_COUNT_ASPECT
        ).to(Config.DEVICE)

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        model.apply(init_weights)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
        scaler = torch.amp.GradScaler(enabled=(Config.USE_AMP and Config.DEVICE.type == 'cuda'))

        # Get propagation steps for CURRENT valid labels
        aspect_prop_steps = get_aspect_propagation_steps(go_dag, target_labels)

        # Names for saving
        MODEL_FILENAME = Config.MODEL_SAVE_PATH.replace(".pth", f"_{aspect}.pth")
        PRED_FILENAME = Config.PREDICTIONS_MEMMAP_PATH.replace(".npy", f"_{aspect}.npy")

        writer = SummaryWriter(log_dir=f"{Config.LOG_DIR}_{aspect}")

        # --- D. TRAINING LOOP ---
        best_val_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
            
            if Config.DEVICE.type == "cuda": torch.cuda.empty_cache()

            best_t, best_f1 = validate_cafa_pk(
                model, val_loader, ia_weights_aspect, aspect_prop_steps, 
                LABEL_COUNT_ASPECT, Config.DEVICE
            )

            logging.info(f"Ep {epoch} | {aspect} | Loss: {epoch_loss:.4f} | Fmax: {best_f1:.4f} (Th:{best_t:.2f})")
            
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            writer.add_scalar("Metrics/Fmax", best_f1, epoch)
            
            scheduler.step()

            if best_f1 > best_val_f1 + 1e-4:
                best_val_f1 = best_f1
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, best_t, best_f1, MODEL_FILENAME)
            else:
                epochs_no_improve += 1
                logging.info(f"  No imp ({epochs_no_improve}/{Config.PATIENCE})")

            if epochs_no_improve >= Config.PATIENCE:
                logging.info(f"Early stopping {aspect}.")
                break
        
        writer.close()

        # --- E. SAVE PREDICTIONS ---
        save_propagated_predictions(
            model, val_loader, len(val_ds), LABEL_COUNT_ASPECT, 
            aspect_prop_steps, PRED_FILENAME, MODEL_FILENAME
        )
        
        # Clean up
        del model, optimizer, scaler, train_loader, val_loader
        gc.collect()
        if Config.DEVICE.type == "cuda": torch.cuda.empty_cache()

    logging.info("ALL ASPECTS FINISHED.")
    logging.info("Run the final aggregation script to combine predictions from all aspects.")