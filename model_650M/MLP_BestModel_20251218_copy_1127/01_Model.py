import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import TrainConfig as Config
from utils import (
    ResidualMLP, load_and_preprocess_data, calculate_pos_weight, SparseLabelDataset, 
    save_checkpoint, save_predictions_to_memmap,
    set_seed, SummaryWriter, ModelEMA, ModelCheckpointer, compute_fmax
)

if __name__ == "__main__":
    print(f"Device: {Config.DEVICE}")
    set_seed()
    
    # 1. Data Loading
    X_train, X_val, Y_train_sparse, Y_val_sparse, INPUT_DIM, LABEL_COUNT = load_and_preprocess_data()
    N_train = len(X_train)

    # 2. Pos Weight & Criterion
    pos_weight_np = calculate_pos_weight(Y_train_sparse, N_train)
    pos_weight_torch = torch.tensor(pos_weight_np, dtype=torch.float32).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_torch)

    # 3. Datasets & Loaders
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

    # 4. Model & EMA
    full_dims = [INPUT_DIM] + Config.HIDDEN_DIMS 
    model = ResidualMLP(
        full_dims, 
        dropout=Config.DROPOUT_RATE, 
        output_dim=LABEL_COUNT
    ).to(Config.DEVICE)

    ema = ModelEMA(model, decay=0.999) 

    # 5. Optimizer, Scaler, and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scaler = torch.amp.GradScaler(enabled=(Config.USE_AMP and Config.DEVICE.type == 'cuda'))

    total_steps = len(train_loader) * Config.NUM_EPOCHS
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    # Fix: Use LambdaLR for your custom lr_lambda logic
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    checkpointer = ModelCheckpointer(Path(f"runs/{Config.EXP_NAME}/checkpoints"), k=3)
    writer = SummaryWriter(log_dir=Config.LOG_DIR)

    # 6. Training Loop
    print("\n--- Starting Training Loop ---")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        # Inner loop: This is where the actual training happens
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for xb, yb in pbar:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            
            optimizer.zero_grad()

            # Forward pass with Autocast
            with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                logits = model(xb)
                loss = criterion(logits, yb)

            # Backward pass with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Step Scheduler per batch (required for warmup)
            scheduler.step()
            
            # Update EMA weights
            ema.update(model)
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)

        # 7. Validation
        model.eval()
        with torch.no_grad():
            # Using your provided compute_fmax from utils
            val_fmax = compute_fmax(model, val_loader, Config.DEVICE)
            ema_fmax = compute_fmax(ema.module, val_loader, Config.DEVICE)

        print(f"Epoch {epoch} | Loss: {avg_loss:.5f} | Val Fmax: {val_fmax:.5f} | EMA Fmax: {ema_fmax:.5f}")

        # 8. Logging & Checkpointing
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Metrics/Val_Fmax", val_fmax, epoch)
        writer.add_scalar("Metrics/EMA_Fmax", ema_fmax, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # Checkpoint the EMA model (standard practice in CAFA)
        checkpointer.checkpoint(ema.module, ema_fmax, epoch)

        if Config.DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    writer.close()
    
    # Final Prediction Saving using the best EMA model
    print("Saving final predictions...")
    save_predictions_to_memmap(ema.module, val_loader, len(val_ds), LABEL_COUNT)