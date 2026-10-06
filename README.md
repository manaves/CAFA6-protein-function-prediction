# CAFA6 Protein Function Prediction

Predict Gene Ontology (GO) terms for protein sequences for the CAFA6 challenge.

This code was developed for the Kaggle competition
[CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction),
where it finished **1407th out of 2177 teams**.

The pipeline turns raw protein sequences into ESM-2 650M embeddings, trains a
residual MLP to score GO terms, and applies hierarchical GO propagation. Models
are selected with the CAFA protein-centric weighted F-max (CAFA-PK).

## Repository layout

```
.
├── 01_data_preparation/          # Step 1 – data exploration and preparation
│   ├── 00_DataExploration.ipynb
│   └── 01_DataPreparation.ipynb
├── model_650M/                   # ESM-2 650M embeddings + MLP model
│   ├── 02_embeddings/            # Step 2 – generate embeddings and labels
│   ├── 03_fold_split/            # Step 3 – cross-validation fold split
│   ├── 04_mlp_model/             # Step 4 – training, evaluation and submission
│   ├── input/                    # generated features/arrays    (git-ignored)
│   └── output/                   # checkpoints/predictions      (git-ignored)
├── general_input/                # shared raw and prepared inputs
├── archive/                      # archived earlier experiments  (git-ignored)
├── requirements.txt
└── README.md
```

> The pipeline folders are prefixed with their execution order. Steps 2–4 live
> under `model_650M/` because they operate on that model's data and artifacts.
> All paths inside the notebooks and scripts are relative to their own folder,
> so the code can be run from any working directory.

## Execution order

Run the stages in order. Each stage consumes the outputs of the previous one;
see the inputs/outputs listed below.

### Step 1 — Data preparation (`01_data_preparation/`)

Exploratory analysis and preparation of the training table shared by all models.

| Notebook | Purpose | Output |
| --- | --- | --- |
| `00_DataExploration.ipynb` | EDA: sequence-length distribution, most frequent GO terms and taxa (train and test). Read-only. | — |
| `01_DataPreparation.ipynb` | Parse FASTA headers and merge GO annotations into a single table. | `general_input/train/train_data_prepared.csv` |

### Step 2 — Embeddings and labels (`model_650M/02_embeddings/`)

Generate ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) embeddings and the
single split with its sparse labels and IA weights.

| Notebook | Purpose | Output |
| --- | --- | --- |
| `01_Embeddings.ipynb` | Embed the training sequences. | `model_650M/input/train_embeddings.npy` |
| `02_EmbeddingsTest.ipynb` | Embed the test sequences. | `model_650M/input/test_embeddings.npy` |
| `03_DataSplit.ipynb` | Split train/val and slice the embeddings. | `input/train/train_data_split.csv`, `input/val/val_data_split.csv`, `input/train/train_embeddings_split.npy`, `input/val/val_embeddings_split.npy` |
| `04_OneHotEncoding.ipynb` | Encode GO labels (sparse) and build classes and IA weights. | `input/train/Y_train_sparse.npy`, `input/val/Y_val_sparse.npy`, `input/classes.npy`, `input/ia_weights.npy` |

### Step 3 — Cross-validation fold split (`model_650M/03_fold_split/`)

Create the fold-0 split used by the final model and its labels/weights.

| Notebook | Purpose | Output |
| --- | --- | --- |
| `01_FoldSplit.ipynb` | Build the fold-0 train/val split from the cluster assignment and slice the embeddings. | `input/train/train_fold0_split.csv`, `input/val/val_fold0_split.csv`, `input/train/train_fold0_embeddings.npy`, `input/val/val_fold0_embeddings.npy` |
| `02_OneHotEncoding_Fold.ipynb` | Encode fold-0 GO labels and build fold classes and IA weights. | `input/train/Y_train_fold0_sparse.npy`, `input/val/Y_val_fold0_sparse.npy`, `input/classes_fold0.npy`, `input/ia_weights_fold0.npy` |

### Step 4 — MLP model (`model_650M/04_mlp_model/`)

Train the model on the fold-0 features and produce the submission.

| Script | Purpose |
| --- | --- |
| `00_LRGrid.py` | Quick learning-rate sweep (`Config.SWEEP_EPOCHS` epochs per LR). |
| `01_Model.py` | Full training with early stopping; saves the best checkpoint and propagated validation predictions. |
| `02_Submission.py` | Test inference, GO propagation and submission file. |
| `config.py` | All paths and hyperparameters. |
| `utils.py` | Data loading, model, GO DAG/propagation and CAFA-PK metric. |

Key outputs (`model_650M/output/`): `best_mlp_model_fold0_2.pth`,
`val_predictions_propagated_fold0_2.npy`, `val_predictions_with_ids.tsv`,
`submission.tsv`. TensorBoard logs and training logs go to `model_650M/04_mlp_model/runs/`.

Run a script from anywhere, for example:

```bash
python model_650M/04_mlp_model/01_Model.py
```

## Model summary

- **Backbone features:** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) embeddings.
- **Classifier:** residual MLP, hidden dims `[1024, 1024, 512]`, dropout `0.4`.
- **Loss:** `BCEWithLogitsLoss` with positive-class weighting (optional label smoothing).
- **Optimizer:** AdamW (`weight_decay = 5e-3`) with cosine annealing and mixed precision.
- **Evaluation:** CAFA protein-centric weighted F-max (CAFA-PK) with GO propagation.
- **Reported model:** fold 0.

Main hyperparameters live in `model_650M/04_mlp_model/config.py`.

## Data

- `general_input/` — shared inputs: `train/train_sequences.fasta`,
  `train/train_terms.tsv`, `train/train_taxonomy.tsv`, `train/go-basic.obo`,
  `train/train_data_prepared.csv`, `IA.tsv`, `test/testsuperset.fasta`,
  `test/testsuperset-taxon-list.tsv`.
- `model_650M/input/` — generated features, labels and weights (git-ignored).
  Regenerate it by running Steps 2–3.
- `model_650M/output/` — model checkpoints and predictions (git-ignored).

## Setup

The development environment is captured in `requirements.txt` (a `conda list`
export). The main direct dependencies are: `torch`, `transformers`, `numpy`,
`pandas`, `scipy`, `scikit-learn`, `biopython`, `obonet`, `networkx`,
`goatools`, `tqdm` and `tensorboard`. A CUDA-capable GPU is strongly recommended
for embedding generation and training.

## Notes

- `model_650M/input/`, `model_650M/output/`, `model_650M/04_mlp_model/runs/`
  and `archive/` are git-ignored and are not part of the repository.
- `archive/` holds earlier experiments that are no longer part of the pipeline.
