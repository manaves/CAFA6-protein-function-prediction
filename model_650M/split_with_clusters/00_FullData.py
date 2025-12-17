import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import os

# --- Configuration ---
GENERAL_INPUT_PATH = "../../general_input/"
INPUT_PATH = "../input/"
TRAIN_PATH = os.path.join(INPUT_PATH, "train/")
Y_FULL_SPARSE_PATH = os.path.join(TRAIN_PATH, "Y_sparse_all_train.npy")
IA_PATH = os.path.join(GENERAL_INPUT_PATH, "IA.tsv")
ALL_SEQUENCE_IDS_PATH = os.path.join(INPUT_PATH, "sequences_ids_all_train.npy")
CLASSES_PATH = os.path.join(INPUT_PATH, "classes_all_train.npy")
IA_WEIGHTS_PATH = os.path.join(INPUT_PATH, "ia_weights_all_train.npy")

GENERAL_INPUT_PATH = "../../general_input/"
TRAIN_CSV_PATH = os.path.join(GENERAL_INPUT_PATH, "train/train_data_prepared.csv")

def safe_eval_and_flatten(x):
    if pd.notna(x) and x.strip().startswith('['):
        try:
            # 1. Convert string to list of lists (e.g., "[['GO:1'], ['GO:2', 'GO:3']]")
            evaluated_list_of_lists = ast.literal_eval(x)
            
            # 2. Flatten the list of lists into a single list
            return [term for sublist in evaluated_list_of_lists for term in sublist]
        except:
            # Return empty list if evaluation fails
            return []
    return []

def parse_go_terms_flattened(go_terms_series):
    # Apply the safe evaluation and flattening function to the series
    return go_terms_series.apply(safe_eval_and_flatten)

if __name__ == "__main__":
    
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    # Save the ids in a numpy array
    all_sequence_ids = df_train["id"].unique().tolist()
    np.save(ALL_SEQUENCE_IDS_PATH, all_sequence_ids)

    y_train_lists = parse_go_terms_flattened(df_train["go_terms"])
    # Binarize the go terms
    mlb = MultiLabelBinarizer()
    # Fit the multilabel binarizer and transform the go terms
    Y_full_sparse = mlb.fit_transform(y_train_lists)
    # Get the classes
    classes = mlb.classes_
    
    # Check if the number of classes matches the number of columns in the sparse matrix
    if len(classes) == Y_full_sparse.shape[1]:
        print("OK! The number of classes matches the number of columns in the sparse matrix.")
        # Save the classes
        np.save(CLASSES_PATH, classes)
    else:
        print("WARNING: The number of classes does not match the number of columns in the sparse matrix.")
        print(f"Number of classes: {len(classes)}")
        print(f"Number of columns in the sparse matrix: {Y_full_sparse.shape[1]}")
        raise ValueError("The number of classes does not match the number of columns in the sparse matrix.")
    
    
    # Load the IA weights
    df_weights = pd.read_csv(IA_PATH, sep="\t", names=["go_terms", "ia"])
    ia_dict = dict(zip(df_weights["go_terms"], df_weights["ia"]))
    ia_weights = np.zeros(len(classes), dtype=np.float32)
    for i, term in enumerate(classes):
        ia_weights[i] = ia_dict.get(term, 0.0)
    # Save the IA weights
    np.save(IA_WEIGHTS_PATH, ia_weights)
    
    # Save the sparse matrix
    np.save(Y_FULL_SPARSE_PATH, Y_full_sparse)