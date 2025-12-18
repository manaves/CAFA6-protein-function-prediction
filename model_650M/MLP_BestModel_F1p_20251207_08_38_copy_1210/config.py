import os
import time
import torch

# --- CONFIGURATION PARAMETERS ---
class Config:
    """A class to hold all configuration parameters."""
    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")
    GENERAL_INPUT_DIR = "../../general_input/"

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_embeddings_split_prop.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_embeddings_split_prop.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse_prop.npy")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse_prop.npy")
    IA_WEIGHTS_PATH = os.path.join(INPUT_DIR, "ia_weights_prop.npy")
    OBO_FILE_PATH = os.path.join(GENERAL_INPUT_DIR, "train/go-basic.obo")
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes_prop.npy")
    
    # Log paths
    LOG_DIR = os.path.join("./runs", "mlp_best_model_" + time.strftime("%Y%m%d_%H_%M"))
    LOG_FILE = "train_log.txt"
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "LR_sweep_log.txt")

    # Output Paths
    # IMPORTANT: Change PREDICTIONS_MEMMAP_PATH to clearly indicate PROPAGATED predictions
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_predictions_propagated_prop.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_prop.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 300
    PATIENCE = 25
    HIDDEN_DIMS = [1024, 1024, 512]
    DROPOUT_RATE = 0.4
    LABEL_SMOOTHING_EPSILON = 0.0
    
    # --- SWEEP-SPECIFIC PARAMETERS ---
    SWEEP_EPOCHS = 5     # Number of epochs to run for the quick sweep
    # -----------------------------------

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True