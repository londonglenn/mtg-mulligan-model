# MTG Mulligan Modeling Pipeline

This repository trains, evaluates, compares, and publishes machine learning models for predicting *Magic: The Gathering* mulligan decisions.

The project models the probability that a player keeps a seven-card opening hand using hand composition, play/draw context, land count features, card identity features, and mana value features.

## Project Overview

The pipeline supports:

- Downloading collected mulligan data from the server
- Cleaning and preprocessing raw hand-decision files
- Building reproducible dataset snapshots
- Training multiple model types
- Running feature ablation studies
- Evaluating models with cross-validation
- Saving model artifacts, metrics, predictions, and feature reports
- Publishing a trained model bundle to the server
- Loading the latest model for prediction

## Repository Structure

```text

├── data/
│   ├── raw/                  # Raw uploaded CSV / Excel files
│   └── processed/
│       ├── card_info_cache.json
│       └── datasets/         # Versioned processed datasets
│
├── models/
│   ├── registry.json         # Model run registry
│   ├── latest.json           # Latest/default model run
│   └── runs/                 # One folder per trained run
│
├── reports/
│   └── ablation_analysis/    # Ablation summary tables and figures
│
├── analyze_runs.py           # Builds comparison plots/tables from recent runs
├── evaluate.py               # Cross-validates trained models and saves reports
├── features.py               # Shared feature engineering logic
├── ingest.py                 # Downloads and deduplicates uploaded data
├── pipeline.py               # End-to-end pipeline runner
├── predict.py                # Loads a trained model and predicts one hand
├── preprocess.py             # Cleans raw data and builds dataset snapshots
├── publish_model.py          # Uploads model bundles to the server
├── registry.py               # Tracks model runs and latest model
├── train.py                  # Trains/tunes models
└── utils.py                  # Shared filesystem, JSON, ID, and hashing helpers
```
Here is your content cleaned up into a structured, paste-ready Markdown README format:

---

# MTG Mulligan Model Pipeline

## Setup

### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib requests matplotlib
```

---

## Pipeline Usage

Run the full pipeline:

```bash
python pipeline.py
```

You will be prompted to choose:

* **Single full-feature study**
* **Full ablation study**

---

## Modes

### Mode 1: Single Full-Feature Study

Runs all supported models on the full feature set:

* Logistic Regression (L1 regularization)
* XGBoost
* LightGBM

---

### Mode 2: Full Ablation Study

Runs each model across different feature configurations:

* `full`
* `no_mana_buckets`
* `land_only`
* `cards_only`

This allows comparison of how much performance comes from:

* Land features
* Card identity features
* Mana curve features

---

## Manual Step-by-Step Usage

### 1. Ingest Data

```bash
python ingest.py
```

* Downloads uploaded data from the server
* Extracts the archive
* Copies only new CSV/Excel files into:

  ```
  data/raw/
  ```

---

### 2. Preprocess Data

```bash
python preprocess.py
```

Creates processed dataset:

```
data/processed/datasets/<dataset_id>/mulligan_data.csv
```

#### Preprocessing Includes:

* Normalize keep/mulligan labels
* Normalize play/draw context
* Deduplicate rows
* Build step-encoded card features
* Fetch Scryfall metadata
* Add land-count features
* Add mana-value bucket features

---

### 3. Train Models

Usually run via `pipeline.py`, but can be called directly:

```python
import train

run_id = train.main(
    dataset_id="YOUR_DATASET_ID",
    experiment_id="full",
    model_type="logreg_l1",
)
```

#### Supported Models:

* `logreg_l1`
* `xgboost`
* `lightgbm`

#### Supported Experiments:

* `full`
* `no_mana_buckets`
* `land_only`
* `cards_only`

---

### 4. Evaluate a Run

```python
import evaluate

evaluate.main(run_id="YOUR_RUN_ID")
```

#### Evaluation Outputs:

* Cross-validated accuracy
* ROC-AUC
* Log loss
* Out-of-fold log loss
* Threshold sweep
* Best balanced-accuracy threshold
* Confusion matrix
* Prediction-level results
* Misclassified hands:

  * False keeps
  * False mulligans
* Top feature reports

---

### 5. Analyze Recent Runs

```bash
python analyze_runs.py
```

Outputs:

```
reports/ablation_analysis/
reports/ablation_analysis/figures/
```

Includes:

* Ablation tables
* Performance comparison plots

---

### 6. Predict a Single Hand

```python
from predict import predict

hand = [
    "Ragavan, Nimble Pilferer",
    "Guide of Souls",
    "Ocelot Pride",
    "Arid Mesa",
    "Sacred Foundry",
    "Galvanic Discharge",
    "Phlage, Titan of Fire's Fury",
]

result = predict(hand, on_play=1)

print(result)
```

#### Example Output:

```json
{
  "keep_probability": 0.73,
  "decision": 1
}
```

* `decision = 1` → Keep
* `decision = 0` → Mulligan

---

## Model Artifacts

Each run is saved under:

```
models/runs/<run_id>/
```

### Contents:

* `model.pkl`
* `feature_columns.json`
* `metadata.json`
* `metrics.json`
* `threshold_sweep.csv`
* `confusion_matrix.csv`
* `predictions.csv`
* `misclassified.csv`
* `false_keeps.csv`
* `false_mulligans.csv`
* `top_features.csv`
* `selected_features.csv`
* `run_summary.txt`

---

## Model Registry

* Registry:

  ```
  models/registry.json
  ```
* Latest model:

  ```
  models/latest.json
  ```

---

## Publishing a Model

### 1. Set Admin Token

```powershell
$env:ADMIN_TOKEN="your_admin_token_here"
```

### 2. Publish

```python
import publish_model

publish_model.main(run_id="YOUR_RUN_ID", set_latest=True)
```

### Published Bundle Includes:

* Trained model
* Feature columns
* Metadata
* Metrics
* SHA-256 manifest

---

## Feature Engineering

Feature logic is defined in:

```
features.py
```

### Feature Groups

#### Card Identity Features

Step-encoded representation:

* `CardName_1`
* `CardName_2`

---

#### Land Features

* `num_lands`
* `num_lands_sq`

Captures nonlinear land curve behavior.

---

#### Mana Value Buckets

* `0_drops`
* `1_drops`
* `2_drops`
* `3_drops`
* `4_drops`
* `5_drops`
* `6_plus_drops`

> Lands are excluded.

---

#### Play/Draw Context

* `on_play = 1` → on the play
* `on_play = 0` → on the draw

---

## Evaluation Metrics

### ROC-AUC

* Measures ranking quality across thresholds

---

### Log Loss (Primary Metric)

* Measures probability calibration
* Lower is better

---

### Accuracy

* Fraction of correct predictions at a threshold

---

### Balanced Accuracy

* Averages performance across both classes
* Useful for class imbalance

---

## Notes

* Training uses randomized hyperparameter search (optimized for log loss)
* Evaluation uses 5-fold stratified cross-validation
* LightGBM uses feature-name mapping for compatibility
* Scryfall metadata is cached locally
* Dataset and run IDs are auto-generated for reproducibility