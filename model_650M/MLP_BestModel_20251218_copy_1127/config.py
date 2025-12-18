import os
import time
import torch


class TrainConfig:
    """Configuration parameters for training the MLP model."""

    # File Paths
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = os.path.join(INPUT_DIR, "val/")
    INPUT_TRAIN_DIR = os.path.join(INPUT_DIR, "train/")

    # Input Data Paths
    TRAIN_EMB_PATH = os.path.join(INPUT_TRAIN_DIR, "train_fold0_embeddings.npy")
    VAL_EMB_PATH = os.path.join(INPUT_VAL_DIR, "val_fold0_embeddings.npy")
    Y_TRAIN_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_fold0_sparse.npz")
    Y_VAL_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_fold0_sparse.npz")

    # Log path
    LOG_DIR = os.path.join("./runs", "mlp_best_model_" + time.strftime("%Y%m%d_%H_%M"))
    LOG_FILE = "train_log.txt"

    # Output Paths
    PREDICTIONS_MEMMAP_PATH = os.path.join(OUTPUT_DIR, "val_fold0_predictions.npy")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_fold0.pth")

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-3
    NUM_EPOCHS = 200
    PATIENCE = 15  # Early stopping patience
    EMBEDDING_DIM = 1280
    HIDDEN_DIMS = [1024, 1024, 512]
    DROPOUT_RATE = 0.4  # Best score with 0.4
    LABEL_SMOOTHING_EPSILON = 0.0

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True  # Automatic Mixed Precision (AMP)


class InferenceConfig:
    """Configuration parameters for inference and submission generation."""

    # Base Paths
    GENERAL_INPUT_DIR = "../../general_input/"
    INPUT_DIR = "../input/"
    OUTPUT_DIR = "./output/"

    # Input Files
    EMBEDDINGS_PATH = os.path.join(INPUT_DIR, "test_embeddings.npy")
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes_all_train.npy")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model.pth")
    OBO_PATH = os.path.join(GENERAL_INPUT_DIR, "train/go-basic.obo")
    FASTA_PATH = os.path.join(GENERAL_INPUT_DIR, "test/testsuperset.fasta")

    # Output Files
    RAW_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_raw.npy")
    PROP_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_propagated.npy")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.tsv")

    # Inference Parameters
    BATCH_SIZE = 1024
    THRESHOLD = 0.55
    DROPOUT = 0.4  # Must match training dropout rate

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


