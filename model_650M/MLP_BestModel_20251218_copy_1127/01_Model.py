import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import TrainConfig as Config
from utils import (
    ResidualMLP, load_and_preprocess_data, calculate_pos_weight, SparseLabelDataset, 
    train_one_epoch, validate_epoch, save_checkpoint, save_predictions_to_memmap,
    set_seed, SummaryWriter
)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Device: {Config.DEVICE}")
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
