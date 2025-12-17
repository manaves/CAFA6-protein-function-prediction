import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils import (
    ResidualMLP,
)


# =========================
# CONFIG
# =========================
DEVICE = Config.DEVICE
BATCH_SIZE = Config.BATCH_SIZE

SUBMISSION_FILE = Config.SUBMISSION_PATH  # e.g. "submission.tsv"

ASPECTS = ["BP", "MF", "CC"]


# =========================
# DATASET
# =========================
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, X, protein_ids):
        self.X = X
        self.protein_ids = protein_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.protein_ids[idx]


def load_data_and_model() -> tuple:
    """Loads FASTA IDs, embeddings, model, and calculates dimensions."""
    print("## Loading Data and Model")
    
    # Read FASTA IDs
    test_ids = []
    with open(Config.FASTA_PATH, 'r') as f:
        for line in tqdm(f, desc="Reading FASTA IDs"):
            if line.startswith(">"):
                test_ids.append(line.strip().split()[0][1:])

    # Load Embeddings (using mmap for efficiency)
    emb_test = np.load(Config.TEST_EMB_PATH, mmap_mode="r")
    real_samples = emb_test.shape[0]

    # Align samples and IDs
    NUM_SAMPLES = len(test_ids)
    if real_samples != NUM_SAMPLES:
        print(f"WARNING: Sample count mismatch ({real_samples} in file vs {NUM_SAMPLES} in FASTA). Truncating.")
        NUM_SAMPLES = min(real_samples, NUM_SAMPLES)
        test_ids = test_ids[:NUM_SAMPLES]

    EMBEDDING_DIM = emb_test.shape[1]
    
    print(f"Total samples to process: {NUM_SAMPLES}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    # Load Model and Checkpoint
    model_classes = np.load(Config.CLASSES_PATH)
    NUM_CLASSES = len(model_classes)
    
    
    return emb_test, test_ids, model_classes, NUM_SAMPLES, NUM_CLASSES

# =========================
# LOAD TEST DATA
# =========================
print("Loading test data...")
X_test, protein_ids, model_classes, NUM_SAMPLES, NUM_CLASSES = load_data_and_model()

INPUT_DIM = X_test.shape[1]
# Load GO terms as a Python list so we can use .index()
ALL_GO_TERMS_LIST = np.load(Config.CLASSES_PATH, allow_pickle=True).tolist()

test_ds = TestDataset(X_test, protein_ids)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print(f"Test samples: {len(test_ds)}")
print(f"Total GO terms: {len(ALL_GO_TERMS_LIST)}")


# =========================
# LOAD GO → ASPECT MAP
# =========================
df_terms = pd.read_csv(
    Config.TRAIN_TERMS_PATH,
    sep="\t",
    names=["protein", "go_term", "aspect"],
    skiprows=1,
)

term2aspect = (
    df_terms[["go_term", "aspect"]]
    .drop_duplicates()
    .set_index("go_term")["aspect"]
    .to_dict()
)

ASPECT_LABELS = {
    "BP": [t for t in ALL_GO_TERMS_LIST if term2aspect.get(t) == "P"],
    "MF": [t for t in ALL_GO_TERMS_LIST if term2aspect.get(t) == "F"],
    "CC": [t for t in ALL_GO_TERMS_LIST if term2aspect.get(t) == "C"],
}


# =========================
# OPEN SUBMISSION FILE (CAFA6 / Kaggle format)
# =========================
os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

# CAFA6 Kaggle expects the header:
# protein_id<TAB>go_term<TAB>prediction
fout = open(SUBMISSION_FILE, "w")
fout.write("protein_id\tgo_term\tprediction\n")


# =========================
# INFERENCE PER ASPECT
# =========================
for aspect in ASPECTS:
    print(f"\n=== Running inference for {aspect} ===")

    aspect_terms = ASPECT_LABELS[aspect]
    # Number of labels for this aspect is simply the length of aspect_terms
    LABEL_COUNT = len(aspect_terms)
    print(f"{aspect} labels: {LABEL_COUNT}")

    # Load model
    model = ResidualMLP(
        [INPUT_DIM] + Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT_RATE,
        output_dim=LABEL_COUNT,
    ).to(DEVICE)

    ckpt_path = Config.MODEL_SAVE_PATH.replace(".pth", f"_{aspect}.pth")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for xb, pids in tqdm(test_loader, desc=f"{aspect} inference"):
            xb = xb.to(DEVICE)

            if Config.USE_AMP and DEVICE.type == "cuda":
                with torch.autocast(device_type=DEVICE.type):
                    logits = model(xb)
            else:
                logits = model(xb)

            scores = torch.sigmoid(logits).cpu().numpy()

            # WRITE DIRECTLY (NO THRESHOLD, NO PROPAGATION)
            for i in range(scores.shape[0]):
                protein_id = pids[i]
                row_scores = scores[i]

                nz = row_scores > 1e-3  # optional pruning (keeps file smaller)
                for j in np.where(nz)[0]:
                    fout.write(
                        f"{protein_id}\t{aspect_terms[j]}\t{row_scores[j]:.6f}\n"
                    )

    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


fout.close()

print("\nSubmission file written to:")
print(SUBMISSION_FILE)
