# Semantic Feature-Group Mask Augmentation for Flow-Based NIDS

Reproducibility package for **“Semantic Feature-Group Mask Augmentation for Flow-Based Intrusion Detection under Unseen Telemetry Loss and Dataset Shift.”**

The study trains on CICIDS2017 and performs a final zero-target evaluation on a sealed CSE-CIC-IDS2018 partition. The intervention creates one additional source-training copy per flow and replaces one randomly selected semantic feature group with source-training medians. The component ablation identifies **mask augmentation without availability indicators** as the parsimonious primary method. The full augmentation-plus-indicators model is retained as an ablation component and for the Random Forest model-family sensitivity analysis.

## Main experimental boundary

- Source dataset: CICIDS2017.
- External target: CSE-CIC-IDS2018.
- Shared predictors: 77.
- Semantic groups: 4, containing 32, 23, 12 and 10 predictors.
- Confirmatory seeds: 2027–2031.
- Conditions: complete telemetry, four known single-group losses and six unseen pairwise losses.
- Primary model family: XGBoost.
- Secondary sensitivity analysis: Random Forest.
- Target labels are used only for final evaluation—not for training, calibration, threshold selection, abstention-policy selection or model selection.

The main defensible conclusion is strong source-domain robustness with condition- and model-dependent external transfer. All negative target results are retained.

## Repository structure

```text
config/
  experiment_metadata.json
  feature_renaming_map.json
  semantic_feature_groups.json
environment/
  original_run_environment.json
notebooks/
  authoritative_experiment_notebook.ipynb
protocols/
  replication_protocol_frozen.json
  component_ablation/
  random_forest/
scripts/
  prepare_data.py
  train_xgboost.py
  evaluate_xgboost.py
  frozen_component_ablation.py
  audit_component_ablation_models.py
  prepare_random_forest_replication.py
  train_random_forest_five_seed.py
  evaluate_random_forest_five_seed.py
  aggregate_analysis.py
  generate_figures.py
  verify_repository.py
results/
  xgboost/
  ablation/
  random_forest/
manuscript_tables/
  Table_6.csv ... Table_10.csv
figures/
  Figure_1.pdf ... Figure_5.pdf
```

Raw datasets, frozen Parquet partitions, fitted imputers and model binaries are intentionally excluded because of size and redistribution considerations. Their expected sizes and SHA-256 hashes are recorded in the frozen protocols and audits.

## Fixed partitions

| Partition | Rows | Purpose |
|---|---:|---|
| Source training | 598,043 | Fit models and source medians |
| Source validation | 152,964 | Early stopping |
| Source calibration | 99,084 | Platt calibration and source-only policies |
| Source test | 150,178 | Internal evaluation |
| Sealed external target | 999,143 | Final zero-target evaluation only |

## Reconstruct manuscript tables and figures

No raw data or model training is required for this step.

```bash
python -m pip install -r requirements.txt
python scripts/aggregate_analysis.py
python scripts/generate_figures.py
python scripts/verify_repository.py --require-figures --require-tables
```

The reconstruction script creates Tables 6–10 from the committed seed-level result tables. The figure script creates PDF and 300-dpi PNG versions of the five figures used in the manuscript.

## Full experiment reproduction

Download CICIDS2017 and CSE-CIC-IDS2018 from the Canadian Institute for Cybersecurity. Then prepare the fixed partitions:

```bash
python scripts/prepare_data.py \
  --cicids2017-dir /path/to/CICIDS2017 \
  --cse-cic-ids2018-dir /path/to/CSE-CIC-IDS2018 \
  --output-dir /path/to/NIDS_Research_Checkpoints
```

Select the checkpoint location:

```bash
export NIDS_CHECKPOINT_DIR=/path/to/NIDS_Research_Checkpoints
```

Run the primary XGBoost replication:

```bash
python scripts/train_xgboost.py
python scripts/evaluate_xgboost.py
```

Run the corrected component ablation:

```bash
python scripts/frozen_component_ablation.py
python scripts/audit_component_ablation_models.py
```

Run the Random Forest sensitivity analysis:

```bash
python scripts/prepare_random_forest_replication.py
python scripts/train_random_forest_five_seed.py
python scripts/evaluate_random_forest_five_seed.py
```

The training and evaluation programs use seed-specific completion markers and are restartable.

## Included aggregate results

| Analysis | Non-selective rows | Selective rows | Paired rows |
|---|---:|---:|---:|
| Five-seed XGBoost | 220 | 660 | 110 |
| Corrected four-component ablation | 440 | 1,320 | Not applicable |
| Five-seed Random Forest | 220 | 660 | 110 |

The corrected ablation audit verifies 10 expected trained component models, no missing files, no parameter mismatches, and the SHA-256 identities of both combined result tables. The Random Forest pretraining audit verifies all seven authoritative artifacts and all five fixed Parquet partitions.

This repository is released under the MIT License. Citation metadata is provided in CITATION.cff. Replace the manuscript’s repository URL placeholders with the final GitHub URL before submission.

## Environment record

XGBoost 3.3.0 and CPU execution were recorded for the completed primary experiment. The original Python and scikit-learn versions were not captured in the frozen metadata and are therefore marked as unavailable rather than retrospectively inferred. `requirements.txt` defines a tested-compatible reconstruction environment.

## Data, licensing and citation

This repository does not redistribute CICIDS2017 or CSE-CIC-IDS2018. Users must follow the dataset providers’ terms. This repository is released under the MIT License. Citation metadata is provided in CITATION.cff. Replace the manuscript’s repository URL placeholders with the final GitHub URL before submission.
