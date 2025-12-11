import pandas as pd
import numpy as np
import os
import gc

# --- CONFIGURATION PARAMETERS ---
class Config:
    OUTPUT_DIR = "./output/"
    
    # Input Files
    PREDS_DL = os.path.join(OUTPUT_DIR, "submission_mlp.tsv") 
    PREDS_DIAMOND = os.path.join(OUTPUT_DIR, "submission_diamond.tsv")
    
    # Output File
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.tsv")
    
    # Ensemble Weights (Must sum to 1.0) for when BOTH predict
    WEIGHT_DL = 0.5
    WEIGHT_DIAMOND = 0.5

# --- MAIN FUNCTION ---

def generate_ensemble():
    print("--- Generating Submission Final (Ensemble with Conditional Logic) ---")
    
    # 1. Load Deep Learning Predictions
    print(f"Loading Deep Learning model: {Config.PREDS_DL} ...")
    if os.path.exists(Config.PREDS_DL):
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
    print("Merging predictions...")
    # 'outer': Keep all rows. NaNs will appear where one model didn't predict.
    df_final = pd.merge(df_dl, df_diamond, on=["id", "term"], how="outer")
    
    # Release memory
    del df_dl, df_diamond
    gc.collect()

    # 4. CONDITIONAL SCORING LOGIC
    print("Calculating final scores...")

    # We identify who predicted what using .notna() BEFORE filling NaNs
    has_dl = df_final['score_dl'].notna()
    has_diamond = df_final['score_diamond'].notna()

    # We calculate the weighted average for the case where BOTH exist.
    # We use .fillna(0) inside the calculation just to avoid errors, 
    # but this value will only be used where (has_dl & has_diamond) is True.
    weighted_avg = (df_final['score_dl'].fillna(0) * Config.WEIGHT_DL) + \
                   (df_final['score_diamond'].fillna(0) * Config.WEIGHT_DIAMOND)

    # Define the conditions and choices
    conditions = [
        has_dl & has_diamond,   # Case 1: Both models predicted -> Weighted Average
        has_dl & ~has_diamond,  # Case 2: Only DL predicted -> Trust DL 100%
        ~has_dl & has_diamond   # Case 3: Only Diamond predicted -> Trust Diamond 100%
    ]

    choices = [
        weighted_avg,               # Result for Case 1
        df_final['score_dl'],       # Result for Case 2
        df_final['score_diamond']   # Result for Case 3
    ]

    # np.select applies the logic efficiently
    df_final['score'] = np.select(conditions, choices, default=0.0)

    # 5. Clean and Format
    submission = df_final[['id', 'term', 'score']]
    
    # Sort
    submission = submission.sort_values(by=['id', 'score'], ascending=[True, False])
    
    # Round
    submission['score'] = submission['score'].round(3)

    # 6. Save
    print(f"Saving final file in: {Config.SUBMISSION_FILE}")
    submission.to_csv(Config.SUBMISSION_FILE, sep="\t", index=False, header=False)
    
    print("Process completed successfully!")
    print(f"Total predictions: {len(submission)}")

if __name__ == "__main__":
    generate_ensemble()