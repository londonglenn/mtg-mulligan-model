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
.
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
Setup

Create and activate a virtual environment:

python -m venv .venv

On Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install pandas numpy scikit-learn xgboost lightgbm joblib requests matplotlib
Pipeline Usage

Run the full pipeline:

python pipeline.py

You will be prompted to choose:

1) Single full-feature study
2) Full ablation study
Mode 1: Single Full-Feature Study

Runs all supported models on the full feature set:

Logistic Regression with L1 regularization
XGBoost
LightGBM
Mode 2: Full Ablation Study

Runs each model across these feature configurations:

full
no_mana_buckets
land_only
cards_only

This is useful for comparing how much performance comes from land features, card identity features, and mana curve features.

Manual Step-by-Step Usage
1. Ingest Data
python ingest.py

Downloads the uploaded data zip from the server, extracts it, and copies only new CSV/Excel files into data/raw/.

2. Preprocess Data
python preprocess.py

Creates a processed dataset snapshot under:

data/processed/datasets/<dataset_id>/mulligan_data.csv

The preprocessing step:

Normalizes keep/mulligan labels
Normalizes play/draw context
Deduplicates rows
Builds step-encoded card features
Fetches Scryfall metadata
Adds land-count features
Adds mana-value bucket features
3. Train Models

Training is usually called through pipeline.py, but individual runs can be started from Python:

import train

run_id = train.main(
    dataset_id="YOUR_DATASET_ID",
    experiment_id="full",
    model_type="logreg_l1",
)

Supported model types:

logreg_l1
xgboost
lightgbm

Supported experiment IDs:

full
no_mana_buckets
land_only
cards_only
4. Evaluate a Run
import evaluate

evaluate.main(run_id="YOUR_RUN_ID")

Evaluation saves:

Cross-validated accuracy
Cross-validated ROC-AUC
Cross-validated log loss
Out-of-fold log loss
Threshold sweep
Best balanced-accuracy threshold
Confusion matrix
Prediction-level results
Misclassified hands
False keeps
False mulligans
Top feature reports
5. Analyze Recent Runs
python analyze_runs.py

This generates comparison outputs for recent model runs, including ablation tables and figures.

Outputs are saved to:

reports/ablation_analysis/
figure/
6. Predict a Single Hand
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

Example output:

{
    "keep_probability": 0.73,
    "decision": 1
}

decision = 1 means keep.
decision = 0 means mulligan.

Model Artifacts

Each trained run is saved under:

models/runs/<run_id>/

A typical run folder contains:

model.pkl
feature_columns.json
metadata.json
metrics.json
threshold_sweep.csv
confusion_matrix.csv
predictions.csv
misclassified.csv
false_keeps.csv
false_mulligans.csv
top_features.csv
selected_features.csv
run_summary.txt

The model registry is stored in:

models/registry.json

The default model used by predict.py is tracked in:

models/latest.json
Publishing a Model

To publish a trained model bundle to the server, set your admin token first.

On Windows PowerShell:

$env:ADMIN_TOKEN="your_admin_token_here"

Then run:

import publish_model

publish_model.main(run_id="YOUR_RUN_ID", set_latest=True)

The published bundle includes:

Trained model
Feature column list
Metadata
Metrics
Manifest with SHA-256 hashes
Feature Engineering

The shared feature logic lives in features.py.

Current feature groups include:

Card Identity Features

Cards are represented with step-encoded features.

For example, if a hand contains two copies of a card, both of these features can be active:

Card Name_1
Card Name_2
Land Features
num_lands
num_lands_sq

This lets the logistic model learn that both too few and too many lands can be bad.

Mana Value Buckets
0_drops
1_drops
2_drops
3_drops
4_drops
5_drops
6_plus_drops

Lands are excluded from mana-value buckets.

Play/Draw Context
on_play

on_play = 1 means the hand is on the play.
on_play = 0 means the hand is on the draw.

Evaluation Metrics

The main metrics are:

ROC-AUC

Measures how well the model ranks keepable hands above mulligan hands across all possible thresholds.

Log Loss

Measures the quality of predicted probabilities. This is the main tuning objective.

Lower log loss means the model is not only making correct predictions, but also assigning better-calibrated probabilities.

Accuracy

Measures the fraction of correct keep/mulligan predictions at a chosen threshold.

Balanced Accuracy

Averages performance across both classes. This is useful when the dataset has more keeps than mulligans.

Notes
Training uses randomized hyperparameter search optimized for log loss.
Evaluation uses 5-fold stratified cross-validation.
LightGBM uses a feature-name mapping because some card names contain characters that LightGBM does not accept directly.
Scryfall card metadata is cached locally to avoid repeated API calls.
Dataset and run IDs are generated automatically for reproducibility.