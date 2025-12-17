import pandas as pd
import numpy as np
import ast
import os
from tqdm import tqdm

# --- CONFIGURATION PARAMETERS ---
class Config:
    """Configuration class for file paths and parameters."""
    
    OUTPUT_DIR = "./output/"
    INPUT_DIR = "../input/"
    INPUT_VAL_DIR = INPUT_DIR + "val/"
    INPUT_TRAIN_DIR = INPUT_DIR + "train/"
    
    # Input/Output Files for Class Extraction
    TRAIN_CSV_PATH = os.path.join(INPUT_TRAIN_DIR, "train_data_split.csv")
    SPARSE_MATRIX_PATH = os.path.join(INPUT_TRAIN_DIR, "Y_train_sparse.npy")
    CLASSES_PATH = os.path.join(OUTPUT_DIR, "classes.npy")
    
    # Input Files for Evaluation
    IA_WEIGHTS_PATH = "../../general_input/IA.tsv" # Information Content (IA) weights
    Y_VAL_SPARSE_PATH = os.path.join(INPUT_VAL_DIR, "Y_val_sparse.npy")
    VAL_PREDS_PATH = os.path.join(OUTPUT_DIR, "val_predictions_05_3_propagated.npy")
    
    # Evaluation Parameters
    BATCH_SIZE = 2048
    THRESHOLD_GRID = [
        0.01, 0.02, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    ]
    EPSILON = 1e-8 # Small constant for division safety


# --- CLASS EXTRACTION AND SAFETY CHECK ---

def parse_go_terms_flattened(go_terms_str: str) -> set:
    """Safely parses a string representation of nested GO terms into a flat set."""
    if pd.notna(go_terms_str) and go_terms_str.strip().startswith('['):
        try:
            # Convert string to list of lists, then flatten and return unique terms
            evaluated = ast.literal_eval(go_terms_str)
            return {term for sublist in evaluated for term in sublist}
        except:
            return set()
    return set()

def extract_and_save_classes():
    """Extracts, sorts, saves unique GO classes, and verifies them against the training matrix."""
    print("## GO Class Extraction and Validation")
    print("1. Loading training data to retrieve classes...")
    try:
        df_train = pd.read_csv(Config.TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Training CSV not found at {Config.TRAIN_CSV_PATH}")
        return

    unique_classes = set()

    print("2. Extracting unique GO terms...")
    # Iterate row by row for memory efficiency
    for row in tqdm(df_train['go_terms'], desc="Processing rows"):
        terms = parse_go_terms_flattened(row)
        unique_classes.update(terms)

    # Sort alphabetically (CRUCIAL: Must match MultiLabelBinarizer's internal order)
    sorted_classes = sorted(list(unique_classes))
    classes_array = np.array(sorted_classes)

    print(f"3. Classes recovered: {len(classes_array)}")

    # Safety Check and Saving
    try:
        Y_train_obj = np.load(Config.SPARSE_MATRIX_PATH, allow_pickle=True)
        # Handle case where sparse object is saved in a 0-dim numpy array
        Y_train_sparse = Y_train_obj.item() if Y_train_obj.shape == () else Y_train_obj
        
        num_cols_matrix = Y_train_sparse.shape[1]
        
        if len(classes_array) == num_cols_matrix:
            print("OK! The number of classes matches the trained matrix.")
        else:
            print(f"WARNING: size mismatch.")
            print(f"Unique terms found in CSV: {len(classes_array)}")
            print(f"Columns in sparse matrix: {num_cols_matrix}")
            print("Action required: Check if data filtering was identical.")
            
    except FileNotFoundError:
        print("Sparse matrix not found for verification.")
        
    np.save(Config.CLASSES_PATH, classes_array)
    print(f"File saved to: {Config.CLASSES_PATH}")
    
    return classes_array

# --- WEIGHTED EVALUATION LOGIC ---

def load_ia_weights(classes_array: np.ndarray) -> np.ndarray:
    """Loads Information Content (IA) weights and aligns them with the class array."""
    
    # 1. Load IA Weights
    ia_df = pd.read_csv(
        Config.IA_WEIGHTS_PATH, 
        sep='\t', 
        names=['term', 'ia'], 
        header=None
    )
    ia_dict = dict(zip(ia_df['term'], ia_df['ia']))
    
    # 2. Create aligned weight vector
    weights = np.zeros(len(classes_array), dtype=np.float32)
    for i, term in enumerate(classes_array):
        # If term is not found in IA file, its weight is 0.0 (as per competition rules)
        weights[i] = ia_dict.get(term, 0.0) 
        
    return weights

def calculate_weighted_f1(Y_true_sparse, Y_pred_mmap: np.memmap, weights: np.ndarray):
    """Calculates weighted F1-score for a grid of thresholds."""
    
    num_samples = Y_true_sparse.shape[0]
    num_classes = len(weights)
    batch_size = Config.BATCH_SIZE
    
    # Helper for sigmoid activation
    def sigmoid(x): 
        # Clip input to prevent overflow in exp(-x) for large negative x
        x_clipped = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-x_clipped))

    print("\n## Weighted F1 Evaluation")
    print(f"Total samples: {num_samples}. Total classes: {num_classes}.")
    print("\nResults (Weighted F1):")

    for th in Config.THRESHOLD_GRID:
        w_tp, w_fp, w_fn = 0.0, 0.0, 0.0
        
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            
            # Get Logits/Probs
            logits = Y_pred_mmap[i:end]
            probs = sigmoid(logits)
            
            # Get True Labels
            # Convert sparse matrix slice to dense numpy array
            true_batch = Y_true_sparse[i:end].toarray()
            
            # Binarize Predictions and Convert to Boolean Masks
            preds_bin = (probs >= th)
            true_bin_bool = (true_batch == 1) # Only positive labels are 1

            # Compute Weighted TP, FP, FN
            
            # True Positive (TP): Prediction=1 AND True=1
            tp_mask = preds_bin & true_bin_bool
            w_tp += np.sum(tp_mask * weights) # Apply weight via broadcasting
            
            # False Positive (FP): Prediction=1 AND True=0
            fp_mask = preds_bin & (~true_bin_bool)
            w_fp += np.sum(fp_mask * weights)
            
            # False Negative (FN): Prediction=0 AND True=1
            fn_mask = (~preds_bin) & true_bin_bool
            w_fn += np.sum(fn_mask * weights)
            
        # Calculate Metrics
        
        # Weighted Precision and Recall
        precision = w_tp / (w_tp + w_fp + Config.EPSILON)
        recall = w_tp / (w_tp + w_fn + Config.EPSILON)
        
        # Weighted F1-score
        f1 = 2 * precision * recall / (precision + recall + Config.EPSILON)
        
        print(f"Threshold {th:.2f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # Extract Classes
    classes_array = extract_and_save_classes()
    
    if classes_array is None:
        # Load the saved classes.npy if extraction failed or was skipped
        try:
            classes_array = np.load(Config.CLASSES_PATH)
        except FileNotFoundError:
            print("FATAL ERROR: Could not load or extract class list. Aborting evaluation.")
            exit()
    
    # Setup for Evaluation
    
    # Load IA Weights
    weights_vector = load_ia_weights(classes_array)
    print(f"\nWeights loaded and aligned for {len(weights_vector)} classes.")
    
    # Load True Labels (Sparse)
    Y_true_obj = np.load(Config.Y_VAL_SPARSE_PATH, allow_pickle=True)
    Y_true_sparse = Y_true_obj.item() if Y_true_obj.shape == () else Y_true_obj
    
    # Map Predictions (Memmap)
    num_samples = Y_true_sparse.shape[0]
    Y_pred_mmap = np.memmap(
        Config.VAL_PREDS_PATH, 
        dtype='float32', 
        mode='r', 
        shape=(num_samples, len(classes_array))
    )

    # Run Evaluation
    calculate_weighted_f1(Y_true_sparse, Y_pred_mmap, weights_vector)