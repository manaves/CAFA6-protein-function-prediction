import os
import torch

class Config:
    """A class to hold all configuration parameters."""
    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_embeddings_split.npy")
    TEST_EMB_PATH = os.path.join(INPUT_DIR, "test_embeddings.npy")
    FASTA_PATH = "../../general_input/test/testsuperset.fasta"
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_embeddings_split.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse.npy")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse.npy")

    OBO_FILE_PATH = "../../general_input/train/go-basic.obo" 
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes.npy")
    TRAIN_TERMS_PATH = "../../general_input/train/train_terms.tsv"
    ALL_GO_TERMS_PATH = os.path.join(INPUT_DIR, "classes.npy")  # Same as CLASSES_PATH
    IA_WEIGHTS_PATH = os.path.join(INPUT_DIR, "ia_weights.npy")
    
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.tsv")
    
    # Log path
    LOG_DIR = os.path.join("./runs", "mlp_by_aspect")
    LOG_FILE = "train_log.txt"

    # Output Paths
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_predictions_by_aspect.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_by_aspect.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 300
    PATIENCE = 45  # Early stopping patience
    SWEEP_EPOCHS = 5  # Number of epochs for learning rate sweep
    HIDDEN_DIMS = [1024, 1024, 512]
    DROPOUT_RATE = 0.4 # Best score with 0.4
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True  # Automatic Mixed Precision (AMP)