import torch
import torch.nn as nn
import numpy as np
import obonet
import networkx as nx
import os
import time
from tqdm import tqdm

# --- 1. CONFIGURATION PARAMETERS ---
class Config:
    """Configuration class for file paths and inference parameters."""
    
    # Base Paths
    GENERAL_INPUT_DIR = "../../general_input/"
    INPUT_DIR = "../input/"
    OUTPUT_DIR = "./output/"

    # Input Files
    EMBEDDINGS_PATH = os.path.join(INPUT_DIR, "test_embeddings.npy")
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes.npy")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_mlp_model_05_3.pth")
    OBO_PATH = os.path.join(GENERAL_INPUT_DIR, "train/go-basic.obo")
    FASTA_PATH = os.path.join(GENERAL_INPUT_DIR, "test/testsuperset.fasta")
    
    # Output Files
    RAW_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_raw_05_03.npy")
    PROP_PREDS_PATH = os.path.join(OUTPUT_DIR, "test_preds_propagated_05_03.npy")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.tsv")

    # Inference Parameters
    BATCH_SIZE = 1024
    THRESHOLD = 0.5
    DROPOUT = 0.4 # Must match training dropout rate

    # Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {Config.DEVICE}")
# Ensure numpy load compatibility for dimension check
LABEL_COUNT = np.load(Config.CLASSES_PATH).shape[0]


# --- MODEL DEFINITION (MUST MATCH TRAINING) ---

class ResidualBlock(nn.Module):
    """A compact residual block with Linear-LayerNorm-GELU-Dropout."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies residual connection: output = x + block(x)"""
        return x + self.block(x)
    
class ResidualMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) with a residual architecture."""
    def __init__(self, dims: list, dropout: float, output_dim: int):
        super().__init__()
        layers = []
        
        # Initial projection layer (dims[0] -> dims[1])
        layers.append(nn.Linear(dims[0], dims[1]))
        layers.append(nn.LayerNorm(dims[1]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for i in range(1, len(dims)-1):
            dim_in = dims[i]
            dim_out = dims[i+1]
            
            if dim_in == dim_out:
                layers.append(ResidualBlock(dim_in, dropout))
            else:
                layers.extend([
                    nn.Linear(dim_in, dim_out),
                    nn.LayerNorm(dim_out),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits: head(encoder(x))"""
        return self.head(self.encoder(x))


# --- DATA PREPARATION ---

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
    emb_test = np.load(Config.EMBEDDINGS_PATH, mmap_mode="r")
    real_samples = emb_test.shape[0]

    # Align samples and IDs
    NUM_SAMPLES = len(test_ids)
    if real_samples != NUM_SAMPLES:
        print(f"⚠️ WARNING: Sample count mismatch ({real_samples} in file vs {NUM_SAMPLES} in FASTA). Truncating.")
        NUM_SAMPLES = min(real_samples, NUM_SAMPLES)
        test_ids = test_ids[:NUM_SAMPLES]

    EMBEDDING_DIM = emb_test.shape[1]
    
    print(f"Total samples to process: {NUM_SAMPLES}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")

    # Load Model and Checkpoint
    model_classes = np.load(Config.CLASSES_PATH)
    NUM_CLASSES = len(model_classes)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    # Model initialization
    HIDDEN_DIMS = [EMBEDDING_DIM, 1024, 1024, 512] 
    model = ResidualMLP(HIDDEN_DIMS, dropout=Config.DROPOUT, output_dim=NUM_CLASSES).to(Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Update threshold from checkpoint if available
    threshold = checkpoint.get("threshold", Config.THRESHOLD)
    print(f"Using prediction threshold: {threshold:.4f} (from checkpoint or config)")
    
    return model, emb_test, test_ids, model_classes, NUM_SAMPLES, NUM_CLASSES, threshold

# --- INFERENCE, PROPAGATION, AND SUBMISSION ---

def run_inference(model: nn.Module, emb_test: np.memmap, num_samples: int, num_classes: int):
    """Runs inference on test embeddings and saves raw probabilities to disk."""
    print("## Running Inference")
    
    fp_raw = np.memmap(
        Config.RAW_PREDS_PATH, 
        dtype="float32", 
        mode="w+", 
        shape=(num_samples, num_classes)
    )

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, Config.BATCH_SIZE), desc="Predicting"):
            end = min(i + Config.BATCH_SIZE, num_samples)

            # Load batch from memmap and move to device
            batch = torch.tensor(emb_test[i:end], dtype=torch.float32).to(Config.DEVICE)
            
            # Forward pass: Logits -> Sigmoid -> Probabilities
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Save probabilities to raw memmap
            fp_raw[i:end] = probs

            # Flush periodically
            if i % (Config.BATCH_SIZE * 5) == 0:
                fp_raw.flush()
                time.sleep(0.05)

    fp_raw.flush()
    print(f"Raw inference predictions saved to: {Config.RAW_PREDS_PATH}")

def hierarchical_propagation(num_samples: int, num_classes: int, model_classes: np.ndarray):
    """Applies hierarchical max-propagation (Child score -> Parent score) to the raw predictions."""
    print("## Hierarchical Propagation")
    
    # Build GO DAG (Child -> Parent edges for upward propagation)
    graph = obonet.read_obo(Config.OBO_PATH)
    go_dag = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        if "is_a" in data:
            for parent in data["is_a"]:
                go_dag.add_edge(node, parent)

    # Map terms to column indices
    class_to_idx = {term: i for i, term in enumerate(model_classes)}

    # Determine propagation order (Root to Leaf)
    # The reversed topological sort ensures a term is processed only after all its ancestors 
    # (which might also be children of other terms) have had their scores potentially boosted.
    sorted_terms = [
        term for term in reversed(list(nx.topological_sort(go_dag)))
        if term in class_to_idx
    ]

    # Build propagation steps: (Child Index, Parent Index)
    prop_steps = []
    for child in sorted_terms:
        if child in go_dag:
            c_idx = class_to_idx[child]
            # Successors in our graph are the direct parents
            for parent in go_dag.successors(child):
                if parent in class_to_idx:
                    prop_steps.append((c_idx, class_to_idx[parent]))

    print(f"Propagation steps prepared: {len(prop_steps)}")
    
    # Propagate scores batch-wise
    fp_prop = np.memmap(
        Config.PROP_PREDS_PATH, 
        dtype="float32", 
        mode="w+", 
        shape=(num_samples, num_classes)
    )
    fp_raw_read = np.memmap(
        Config.RAW_PREDS_PATH, 
        dtype="float32", 
        mode="r", 
        shape=(num_samples, num_classes)
    )

    for i in tqdm(range(0, num_samples, Config.BATCH_SIZE), desc="Propagating scores"):
        end = min(i + Config.BATCH_SIZE, num_samples)

        # Load batch copy into RAM
        batch = np.array(fp_raw_read[i:end])

        # Apply propagation: P_parent = max(P_parent, P_child)
        for child_idx, parent_idx in prop_steps:
            batch[:, parent_idx] = np.maximum(
                batch[:, parent_idx], 
                batch[:, child_idx]
            )

        # Write propagated batch to output memmap
        fp_prop[i:end] = batch

        if i % (Config.BATCH_SIZE * 5) == 0:
            fp_prop.flush()
            time.sleep(0.05)

    fp_prop.flush()
    print(f"Propagated predictions saved to: {Config.PROP_PREDS_PATH}")


def write_submission_file(num_samples: int, test_ids: list, model_classes: np.ndarray, threshold: float):
    """Loads final propagated predictions, applies the threshold, and writes the TSV submission file."""
    print(f"## Writing Submission File (Threshold: {threshold:.4f})")
    
    fp_final = np.memmap(
        Config.PROP_PREDS_PATH, 
        dtype="float32", 
        mode="r", 
        shape=(num_samples, len(model_classes))
    )

    with open(Config.SUBMISSION_FILE, "w") as f:
        # Use a reasonable chunk size for writing
        chunk_size = 250 
        for i in tqdm(range(0, num_samples, chunk_size), desc="Writing submission"):
            end = min(i + chunk_size, num_samples)

            batch_probs = np.array(fp_final[i:end])
            batch_ids = test_ids[i:end]

            # Apply threshold and get coordinates (row, column) of positive predictions
            rows, cols = np.where(batch_probs >= threshold)

            buf = []
            # Iterate through positive predictions and format for TSV
            for r, c in zip(rows, cols):
                protein_id = batch_ids[r]
                go_term = model_classes[c]
                confidence_score = batch_probs[r, c]
                buf.append(f"{protein_id}\t{go_term}\t{confidence_score:.3f}\n")

            f.writelines(buf)

            # Flush periodically for safety
            if i % 2500 == 0:
                f.flush()
                os.fsync(f.fileno())
                time.sleep(0.05)

    print(f"Submission saved at: {Config.SUBMISSION_FILE}")


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # Load Data and Model
    model, emb_test, test_ids, model_classes, NUM_SAMPLES, NUM_CLASSES, THRESHOLD = load_data_and_model()
    
    # Run Inference and Save Raw Probabilities
    run_inference(model, emb_test, NUM_SAMPLES, NUM_CLASSES)
    
    # Hierarchical Propagation (Propagate scores, not binary predictions)
    hierarchical_propagation(NUM_SAMPLES, NUM_CLASSES, model_classes)
    
    # Apply Threshold and Write Final Submission File
    write_submission_file(NUM_SAMPLES, test_ids, model_classes, THRESHOLD)