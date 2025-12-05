import os
import networkx as nx
import numpy as np
import obonet
from tqdm import tqdm

# --- CONFIGURATION PARAMETERS ---
class Config:
    """Configuration class for file paths and parameters."""
    
    # Base Directories
    INPUT_DIR = "../input/"
    OUTPUT_DIR = "./output/"
    
    # Input Files
    OBO_FILE_PATH = "../../general_input/train/go-basic.obo"
    CLASSES_PATH = os.path.join(INPUT_DIR, "classes.npy")
    
    # Prediction Files (Input/Output)
    # Using specific files as defined in the original code
    VAL_PREDS_PATH = os.path.join(OUTPUT_DIR, "val_predictions.npy")
    NEW_PREDS_PATH = os.path.join(OUTPUT_DIR, "val_predictions_propagated.npy")
    
    # Propagation Parameters
    BATCH_SIZE = 1024
    FLOAT_DTYPE = 'float32'


# --- CORE FUNCTIONS ---

def load_go_dag(obo_path: str) -> nx.DiGraph:
    """
    Reads the Gene Ontology OBO file and converts it to a NetworkX DiGraph.
    Edges are defined from **child to parent** (for upward propagation).
    """
    print("1. Reading GO file and building DAG...")
    graph = obonet.read_obo(obo_path)
    
    # Initialize a new DiGraph for propagation: Child -> Parent
    go_dag = nx.DiGraph()
    
    for node, data in graph.nodes(data=True):
        go_dag.add_node(node)
        
        # 'is_a' relationships define parent terms
        if 'is_a' in data:
            for parent in data['is_a']:
                # Add edge: Child -> Parent
                go_dag.add_edge(node, parent)
                
    print(f"Graph loaded: {go_dag.number_of_nodes()} nodes.")
    return go_dag

def map_model_classes(classes_path: str, go_dag: nx.DiGraph) -> tuple[dict, list]:
    """
    Loads model class IDs and maps GO terms to their column indices.
    """
    print("2. Mapping classes from the model to GO terms...")
    model_classes = np.load(classes_path)
    
    # Mapping GO Term ID -> Column Index
    class_to_idx = {term: i for i, term in enumerate(model_classes)}

    # Identify which terms from the model are actually in the GO DAG
    valid_terms = [term for term in model_classes if term in go_dag]
    
    print(f"Valid terms found in DAG: {len(valid_terms)} out of {len(model_classes)}")
    return class_to_idx, model_classes

def get_propagation_steps(go_dag: nx.DiGraph, class_to_idx: dict) -> list[tuple[int, int]]:
    """
    Determines the ordered steps (child_idx, parent_idx) needed for score propagation.
    The order ensures child scores are processed before their parents.
    """
    print("3. Establishing topological order for propagation...")
    
    # Topological sort gives nodes in an order where all predecessors (children) of a node appear before the node.
    # Since our graph is Child -> Parent, the order is [Child -> ... -> Parent -> ... -> Root].
    full_topological_order = list(nx.topological_sort(go_dag))
    
    # Filter to only terms in our model and reverse the list.
    # Reversing gives us the order [Root -> ... -> Parent -> ... -> Child].
    # We iterate from Root to Child to ensure the Parent's score is updated *before* the Child's score is used
    # to update the Parent's parent's score (i.e., propagation is done bottom-up).
    # The original code reversed the list: `reversed(full_topological_order)`.
    # Let's verify the original intent:
    # 1. Topological sort (Child -> Parent): [Term-A, Term-B (child of A), ...]
    # 2. Reverse: [Root, ..., Parent, ..., Child]
    # 3. Iterate reversed list (Root to Child). When at `Child`, update `Parent`.
    # This ensures that when we process a child, its parents have already been fully updated by their own children.
    
    sorted_model_terms = [term for term in reversed(full_topological_order) if term in class_to_idx]

    propagation_steps = []
    
    print("Topological order established. Generating propagation steps...")
    
    # Iterate from root-most terms down to leaf terms
    for child in sorted_model_terms:
        child_idx = class_to_idx[child]
        
        # Successors in a Child->Parent graph are the direct parents
        if child in go_dag:
            parents = list(go_dag.successors(child)) 
            for parent in parents:
                if parent in class_to_idx:
                    parent_idx = class_to_idx[parent]
                    # Store the indices: (Child_Index, Parent_Index)
                    propagation_steps.append((child_idx, parent_idx))

    print(f"Parent-child relationships to be processed: {len(propagation_steps)}")
    return propagation_steps

def propagate_predictions_batch(propagation_steps: list[tuple[int, int]], num_samples: int, num_classes: int):
    """
    Performs batch-wise propagation of prediction scores using numpy memmap.
    This is an efficient way to handle large prediction arrays on disk.
    """
    print(f"4. Starting propagation ({num_samples} proteins) batch-wise...")
    
    # Input Memory Map (Read-only)
    Y_input = np.memmap(
        Config.VAL_PREDS_PATH, 
        dtype=Config.FLOAT_DTYPE, 
        mode='r', 
        shape=(num_samples, num_classes)
    )
    
    # Output Memory Map (Write/Update)
    Y_output = np.memmap(
        Config.NEW_PREDS_PATH, 
        dtype=Config.FLOAT_DTYPE, 
        mode='w+', 
        shape=(num_samples, num_classes)
    )

    batch_size = Config.BATCH_SIZE

    for i in tqdm(range(0, num_samples, batch_size)):
        end = min(i + batch_size, num_samples)
        
        # Load batch into memory for processing
        batch_preds = np.array(Y_input[i:end])
        
        # Propagate scores up the hierarchy for this batch
        # parent_score = max(parent_score, child_score)
        for child_idx, parent_idx in propagation_steps:
            batch_preds[:, parent_idx] = np.maximum(
                batch_preds[:, parent_idx], 
                batch_preds[:, child_idx]
            )
            
        # Write the propagated batch back to the output memmap
        Y_output[i:end] = batch_preds

    Y_output.flush() # Ensure data is written to disk
    print(f"Propagation completed. Saved in: {Config.NEW_PREDS_PATH}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Load the GO DAG
    go_dag = load_go_dag(Config.OBO_FILE_PATH)
    
    # Map classes
    class_to_idx, model_classes = map_model_classes(Config.CLASSES_PATH, go_dag)
    num_classes = len(model_classes)
    
    # Determine Propagation Steps
    propagation_steps = get_propagation_steps(go_dag, class_to_idx)
    
    # Determine Input Shape
    # Calculate the number of samples (proteins) based on file size
    # FileSize = num_samples * num_classes * sizeof(float32)
    # sizeof(float32) is 4 bytes
    file_size_bytes = os.path.getsize(Config.VAL_PREDS_PATH)
    num_samples = file_size_bytes // (num_classes * 4)
    
    print(f"Calculated number of samples (proteins): {num_samples}")
    
    # Execute Propagation
    propagate_predictions_batch(propagation_steps, num_samples, num_classes)