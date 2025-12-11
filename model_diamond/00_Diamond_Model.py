import os
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import obonet
from tqdm import tqdm
import sys

# --- CONFIG ---
class Config:
    INPUT_DIR = "../../general_input/"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train/")
    TEST_DIR = os.path.join(INPUT_DIR, "test/")
    OUTPUT_DIR = "./output/"
    
    # Input files
    TRAIN_FASTA = os.path.join(TRAIN_DIR, "train_sequences.fasta")
    TRAIN_TERMS = os.path.join(TRAIN_DIR, "train_terms.tsv")
    TEST_FASTA = os.path.join(TEST_DIR, "testsuperset.fasta")
    OBO_FILE = os.path.join(TRAIN_DIR, "go-basic.obo")
    
    # Output and mid files
    DIAMOND_DB = os.path.join(OUTPUT_DIR, "train_data.dmnd")
    DIAMOND_RESULTS = os.path.join(OUTPUT_DIR, "diamond_matches.tsv")
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_diamond.tsv")
    
    # Diamond parameters
    TOP_K = 10  # K-Nearest Neighbors
    E_VALUE_THRESH = 1e-3 # Quality filter

    # Diamond binary dir
    DIAMOND_CMD = "diamond" 

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# --- DIAMOND FUNCTIONS ---

def run_diamond():
    """Run Diamond: creating DB and align."""
    print("--- 1. Running Diamond Pipeline ---")
    
    # A. Creating DB
    if not os.path.exists(Config.DIAMOND_DB):
        print("Creating Diamond database from train...")
        cmd_db = [
            Config.DIAMOND_CMD, "makedb",
            "--in", Config.TRAIN_FASTA,
            "--db", Config.DIAMOND_DB
        ]
        subprocess.run(cmd_db, check=True)
    else:
        print("Diamond database exist.")

    # B. Align (blastp)
    # Output format 6: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    print("Aligning test vs train sequences...")
    cmd_blast = [
        Config.DIAMOND_CMD, "blastp",
        "--db", Config.DIAMOND_DB,
        "--query", Config.TEST_FASTA,
        "--out", Config.DIAMOND_RESULTS,
        "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore",
        "--max-target-seqs", str(Config.TOP_K),
        "--evalue", str(Config.E_VALUE_THRESH),
        "--sensitive" # Sensitive mode to find far matches
    ]
    
    subprocess.run(cmd_blast, check=True)
    print(f"Alignment finished. Results: {Config.DIAMOND_RESULTS}")

# --- 2. PROCESSING FUNCTIONS ---

def load_train_annotations():
    """Load the actual annotations of train proteins into a dictionary."""
    print("--- 2. Loading training annotations ---")
    df = pd.read_csv(Config.TRAIN_TERMS, sep="\t", names=["id", "term", "aspect"])
    
    # Dict: ProteinID -> Set (GO Terms)
    # We use sets for quick searches.
    annotations = df.groupby("id")["term"].apply(set).to_dict()
    print(f"Annotations loaded for {len(annotations)} proteins.")
    return annotations

def load_ontology():
    """Load the GO graph for propagation."""
    print("--- 3. Loading GO Ontology ---")
    graph = obonet.read_obo(Config.OBO_FILE)
    go_dag = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        go_dag.add_node(node)
        if 'is_a' in data:
            for parent in data['is_a']:
                go_dag.add_edge(node, parent) # Child -> Parent
    return go_dag

def propagate_scores(term_scores, go_dag):
    """
    Propagate scores to parents (Maximum Rule).
    term_scores: Dictionary {GO_Term: Score}
    """
    # We obtain all the terms we have and their ancestors
    propagated = term_scores.copy()
    
    # List with actual terms
    current_terms = list(term_scores.keys())
    
    for term in current_terms:
        score = term_scores[term]
        
        # If the term is in the graph, we propagate upwards.
        if term in go_dag:
            # nx.descendants in obonet (with child->parent edges) returns the PARENTS/ANCESTRAL
            ancestors = nx.descendants(go_dag, term)
            
            for ancestor in ancestors:
                # The ancestor's score is the max(ancestor's current score, child's score).
                old_score = propagated.get(ancestor, 0.0)
                propagated[ancestor] = max(old_score, score)
                
    return propagated

# --- GENERATION OF PREDICTIONS ---

def extract_uniprot_id(diamond_id):
    """
    Extract UniProt ID from Diamond subject ID.
    Diamond returns IDs like 'sp|A0A0C5B5G6|MOTSC_HUMAN'
    We need to extract 'A0A0C5B5G6' to match with train_terms.tsv
    """
    if '|' in diamond_id:
        # Format: sp|UNIPROT_ID|PROTEIN_NAME
        parts = diamond_id.split('|')
        if len(parts) >= 2:
            return parts[1]  # Return the UniProt ID
    # If no pipe, assume it's already the UniProt ID
    return diamond_id

def generate_predictions(train_annotations, go_dag):
    """
    Read Diamond's results and transfer the scores.
    Score = (Percentage of Identity / 100).
    """
    print("--- 4. Generating and Propagating Predictions ---")
    
    # Read Diamond results
    # Columns: qseqid (Test), sseqid (Train), pident (0-100), bitscore
    df_hits = pd.read_csv(Config.DIAMOND_RESULTS, sep="\t", names=["qseqid", "sseqid", "pident", "bitscore"])
    
    # Normalize score to 0-1 (we use pident as a probability proxy)
    df_hits['score'] = df_hits['pident'] / 100.0
    
    # Group hits by query protein (Test)
    # We process protein by protein to save RAM
    queries = df_hits.groupby("qseqid")
    
    final_rows = []
    
    # Buffer for writing in chunks and not saturating memory
    with open(Config.SUBMISSION_FILE, 'w') as f_out:
        
        for query_id, group in tqdm(queries, desc="Procesando proteínas"):
            
            # Temporary dictionary for this protein: {GO_Term: Max_Score}
            protein_preds = {}
            
            # Let's take a look at the hits (neighbors) of this protein.
            for _, row in group.iterrows():
                train_match_id_full = row['sseqid']
                # Extract UniProt ID from Diamond format (sp|ID|NAME -> ID)
                train_match_id = extract_uniprot_id(train_match_id_full)
                match_score = row['score']
                
                # Get the terms that train protein has (Hit)
                if train_match_id in train_annotations:
                    terms = train_annotations[train_match_id]
                    
                    # Transfer terms
                    for term in terms:
                        # If we already have the term, we keep the highest score (Best Hit logic).
                        if term in protein_preds:
                            protein_preds[term] = max(protein_preds[term], match_score)
                        else:
                            protein_preds[term] = match_score
            
            # If there were no hits with runs scored, we skip ahead.
            if not protein_preds:
                continue
                
            # --- PROPAGATION ---
            # We apply the top-down rule in the hierarchy.
            propagated_preds = propagate_scores(protein_preds, go_dag)
            
            # Write to file (Format: ID \t Term \t Score)
            # We filter out very low scores so the file doesn't become huge (optional)
            lines = []
            for term, score in propagated_preds.items():
                if score >= 0.01: # Pequeño filtro de ruido
                    lines.append(f"{query_id}\t{term}\t{score:.3f}\n")
            
            f_out.writelines(lines)

    print(f"Finished. Submission file in: {Config.SUBMISSION_FILE}")

# --- MAIN ---

if __name__ == "__main__":
    # 1. Diamond
    try:
        run_diamond()
    except FileNotFoundError:
        print("ERROR: ‘diamond’ not found. Install it with ‘conda install -c bioconda diamond’ or ‘apt install diamond-aligner’.")
        sys.exit(1)
        
    # 2. Cargar Datos
    train_annots = load_train_annotations()
    dag = load_ontology()
    
    # 3. Generar
    generate_predictions(train_annots, dag)