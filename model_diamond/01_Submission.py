import pandas as pd
import numpy as np
import os
import gc

# --- CONFIGURATION PARAMETERS ---
class Config:
    OUTPUT_DIR = "./output/"
    
    # Input Files
    # 1. NN Predictions (Deep Learning)
    PREDS_DL = os.path.join(OUTPUT_DIR, "submission_mlp.tsv") 
    
    # 2. Diamond predictions (GO_Enrichment)
    PREDS_DIAMOND = os.path.join(OUTPUT_DIR, "submission_diamond.tsv")
    
    # Output File
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.tsv")
    
    # Ensemble Weights (Must sum to 1.0)
    # If you want a simple average, use 0.5 and 0.5
    WEIGHT_DL = 0.7
    WEIGHT_DIAMOND = 0.3

# --- MAIN FUNCTION ---

def generate_ensemble():
    print("--- Generating Submission Final (Ensemble) ---")
    
    # 1. Load Deep Learning Predictions
    print(f"Loading Deep Learning model: {Config.PREDS_DL} ...")
    if os.path.exists(Config.PREDS_DL):
        # We assume it doesn't have a header or it does. 
        # Kaggle usually asks for no header, but pandas handles it better with names.
        # Adjust 'header=None' if your files don't have titles.
        df_dl = pd.read_csv(Config.PREDS_DL, sep="\t", names=["id", "term", "score_dl"])
    else:
        print("WARNING: Deep Learning model not found. Creating empty DataFrame.") 
        df_dl = pd.DataFrame(columns=["id", "term", "score_dl"])

    # 2. Load Diamond Predictions
    print(f"Loading Diamond model: {Config.PREDS_DIAMOND} ...")
    if os.path.exists(Config.PREDS_DIAMOND):
        df_diamond = pd.read_csv(Config.PREDS_DIAMOND, sep="\t", names=["id", "term", "score_diamond"])
    else:
        print("WARNING: Diamond model not found. Using only Deep Learning.")
        df_diamond = pd.DataFrame(columns=["id", "term", "score_diamond"])

    # Memory Optimization: Data Types
    df_dl['score_dl'] = df_dl['score_dl'].astype(np.float32)
    df_diamond['score_diamond'] = df_diamond['score_diamond'].astype(np.float32)

    # 3. Perform Merge (Outer Join)
    print("Merging predictions (this may take a while if the files are large)...")
    # 'outer': Keep all rows from both. Fill with NaN where there is no info.
    df_final = pd.merge(df_dl, df_diamond, on=["id", "term"], how="outer")
    
    # Release memory of the original dataframes
    del df_dl, df_diamond
    gc.collect()

    # 4. Fill NaNs with 0.0
    # If a model didn't predict a term, its probability is 0
    print("Filling missing values with 0.0...")
    df_final['score_dl'] = df_final['score_dl'].fillna(0.0)
    df_final['score_diamond'] = df_final['score_diamond'].fillna(0.0)

    # 5. Calculate the Weighted Average
    print(f"Calculating final score (DL: {Config.WEIGHT_DL}, Diamond: {Config.WEIGHT_DIAMOND})...")
    df_final['score'] = (df_final['score_dl'] * Config.WEIGHT_DL) + \
                        (df_final['score_diamond'] * Config.WEIGHT_DIAMOND)

    # 6. Clean and Format
    # We keep only the necessary columns
    submission = df_final[['id', 'term', 'score']]
    
    # Sort (optional, but it looks better)
    submission = submission.sort_values(by=['id', 'score'], ascending=[True, False])
    
    # Round to 3 decimal places to reduce file size
    submission['score'] = submission['score'].round(3)

    # 7. Save
    print(f"Saving final file in: {Config.SUBMISSION_FILE}")
    # header=False for strict Kaggle format (check CAFA6 rules if they want header or not)
    # Normally CAFA does not want header.
    submission.to_csv(Config.SUBMISSION_FILE, sep="\t", index=False, header=False)
    
    print("Process completed successfully!")
    print(f"Total predictions: {len(submission)}")

if __name__ == "__main__":
    generate_ensemble()