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
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_fold0_embeddings.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_fold0_embeddings.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_fold0_sparse.npy")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_fold0_sparse.npy")
    IA_WEIGHTS_PATH = os.path.join(INPUT_DIR, "ia_weights_fold0.npy")
    OBO_FILE_PATH = os.path.join(GENERAL_INPUT_DIR, "train/go-basic.obo")
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes_fold0.npy")

    # Test Data Paths
    EMBEDDINGS_PATH = os.path.join(INPUT_DIR, "test_embeddings.npy")
    FASTA_PATH = os.path.join(GENERAL_INPUT_DIR, "test/testsuperset.fasta")
    OBO_PATH = os.path.join(GENERAL_INPUT_DIR, "train/go-basic.obo")

    # Output files (submission)
    RAW_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_raw.npy")
    PROP_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_propagated.npy")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.tsv")
    
    # Log paths
    LOG_DIR = os.path.join("./runs", "mlp_best_model_fold0_" + time.strftime("%Y%m%d_%H_%M"))
    LOG_FILE = "train_log.txt"
    LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "LR_sweep_log.txt")

    # Output Paths
    # IMPORTANT: Change PREDICTIONS_MEMMAP_PATH to clearly indicate PROPAGATED predictions
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_predictions_propagated_fold0_2.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_fold0_2.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 300
    PATIENCE = 25
    HIDDEN_DIMS = [1024, 1024, 512]
    DROPOUT_RATE = 0.4
    LABEL_SMOOTHING_EPSILON = 0.0
    BATCH_SIZE_SUBMISSION = 1024

    THRESHOLD = 0.1
    
    # --- SWEEP-SPECIFIC PARAMETERS ---
    SWEEP_EPOCHS = 5     # Number of epochs to run for the quick sweep
    # -----------------------------------

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True