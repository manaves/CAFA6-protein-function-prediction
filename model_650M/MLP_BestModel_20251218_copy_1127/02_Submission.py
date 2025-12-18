import torch
import torch.nn as nn
import numpy as np
import obonet
import networkx as nx
import os
import time
from tqdm import tqdm

from config import InferenceConfig as Config
from utils import load_data_and_model, run_inference, hierarchical_propagation, write_submission_file

# --- 1. CONFIGURATION PARAMETERS ---

print(f"Using device: {Config.DEVICE}")

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