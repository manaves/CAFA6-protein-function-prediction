import pandas as pd
import numpy as np
import os
import scipy.sparse as sp # Needed to handle sparse Y matrix
from sklearn.model_selection import train_test_split

# --- Configuration ---
CLUSTER_DIR = "../../pre"
CLUSTER_TSV_PATH = os.path.join(CLUSTER_DIR, "clust_id25_c80_cov1_mode2_reass1_thr32_cluster.tsv")
RANDOM_SEED = 42
VAL_CLUSTER_RATIO = 0.20 # Percentage of clusters to reserve for validation

INPUT_PATH = "../input/"
TRAIN_PATH = os.path.join(INPUT_PATH, "train/")
VAL_PATH = os.path.join(INPUT_PATH, "val/")
X_FULL_PATH = os.path.join(INPUT_PATH, "train_embeddings.npy")
Y_FULL_SPARSE_PATH = os.path.join(TRAIN_PATH, "Y_sparse_all_train.npz")
ALL_SEQUENCE_IDS_PATH = os.path.join(INPUT_PATH, "sequences_ids_all_train.npy")

X_TRAIN_PATH = os.path.join(TRAIN_PATH, "train_embeddings_new_split.npy")
X_VAL_PATH = os.path.join(VAL_PATH, "val_embeddings_new_split.npy")
Y_TRAIN_PATH = os.path.join(TRAIN_PATH, "Y_train_sparse_new_split.npz")
Y_VAL_PATH = os.path.join(VAL_PATH, "Y_val_sparse_new_split.npz")

def create_homology_split(
    X_full: np.ndarray, 
    Y_full_sparse, 
    all_sequence_ids: list[str], 
    cluster_tsv_path: str, 
    val_ratio: float = 0.20, 
    seed: int = 42
):
    """
    Splits the full dataset (ESM-2 Embeddings X and Labels Y) into train and 
    validation sets based on homology clusters (MMseqs2 output).
    """
    
    # --- 1. Load the Cluster Map ---
    print(f"Loading cluster definitions from: {cluster_tsv_path}")
    df_clusters = pd.read_csv(
        cluster_tsv_path, 
        sep='\t', 
        header=None, 
        names=['Representative_ID', 'Member_ID']
    )
    
    # Map every sequence (Member_ID) to its cluster (Representative_ID)
    sequence_to_cluster_map = df_clusters.set_index('Member_ID')['Representative_ID'].to_dict()
    
    # --- 2. Identify Unique Clusters ---
    cluster_ids = df_clusters['Representative_ID'].unique()
    n_clusters = len(cluster_ids)
    print(f"Total unique clusters found: {n_clusters}")
    
    # --- 3. Split Clusters (e.g., 80/20) ---
    # We split the cluster IDs, not the individual proteins.
    train_clusters, val_clusters = train_test_split(
        cluster_ids, 
        test_size=val_ratio, 
        random_state=seed
    )
    
    print(f"Clusters assigned: Train={len(train_clusters)}, Validation={len(val_clusters)}")

    train_clusters_set = set(train_clusters)
    val_clusters_set = set(val_clusters)
    
    # --- 4. Identify Indices for Split ---
    train_indices = []
    val_indices = []
    
    # Iterate through the main sequence list (which corresponds to rows in X and Y)
    for i, seq_id in enumerate(all_sequence_ids):
        # Look up the cluster ID for the current sequence
        cluster_id = sequence_to_cluster_map.get(seq_id)
        
        if cluster_id in val_clusters_set:
            val_indices.append(i)
        elif cluster_id in train_clusters_set:
            train_indices.append(i)
        else:
            # Handle sequences not found in the cluster map (e.g., if you only clustered a subset)
            # For robustness, we assign them randomly or default them to train.
            # Defaulting to train is often safer to keep the val set "clean"
            if np.random.rand() < val_ratio:
                 val_indices.append(i)
            else:
                 train_indices.append(i)

    print(f"Total samples assigned: Train={len(train_indices)}, Validation={len(val_indices)}")
    print(f"First 10 train indices: {train_indices[:10]}")
    print(f"First 10 val indices: {val_indices[:10]}")
    
    # --- 5. Rebuild Data Matrices ---
    # Use the indices to slice the full X (ESM-2 embeddings) and Y (labels)
    print("Slicing the full data with the train and val indices...")
    X_train = X_full[train_indices, :]
    X_val = X_full[val_indices, :]
    
    # Y matrices (labels)
    print("Slicing the Y matrix with the train and val indices...")
    Y_train = Y_full_sparse[train_indices, :]
    Y_val = Y_full_sparse[val_indices, :]

    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__":
    # Load the full data
    X_full = np.load(X_FULL_PATH)
    Y_full_sparse = sp.load_npz(Y_FULL_SPARSE_PATH)
    all_sequence_ids = np.load(ALL_SEQUENCE_IDS_PATH)

    # Split the data
    X_train, X_val, Y_train, Y_val = create_homology_split(
        X_full=X_full,
        Y_full_sparse=Y_full_sparse,
        all_sequence_ids=all_sequence_ids,
        cluster_tsv_path=CLUSTER_TSV_PATH
    )

    # Save the split data
    print("Saving the split data...")
    np.save(X_TRAIN_PATH, X_train)
    np.save(X_VAL_PATH, X_val)
    sp.save_npz(Y_TRAIN_PATH, Y_train)
    sp.save_npz(Y_VAL_PATH, Y_val)
