# Repository provenance manifest

## Authoritative experiment sources

- `notebooks/authoritative_experiment_notebook.ipynb`: original working notebook.
- `protocols/replication_protocol_frozen.json`: primary five-seed frozen protocol and authoritative artifact hashes.
- `protocols/component_ablation/`: corrected component protocol, completion markers and PASS manifest audit.
- `protocols/random_forest/`: frozen Random Forest protocol, PASS pretraining audit and five-seed evaluation completion marker.
- `scripts/evaluate_xgboost.py`, `scripts/frozen_component_ablation.py` and the Random Forest scripts: restartable execution programs.

## Reconstruction entry points

- `scripts/prepare_data.py`: schema alignment, uniform sampling, hash partitioning, median-imputer fitting and checkpoint writing.
- `config/feature_renaming_map.json`: standalone CSE-CIC-IDS2018-to-CICIDS2017 column mapping.
- `config/semantic_feature_groups.json`: 77-predictor semantic partition.
- `scripts/aggregate_analysis.py`: deterministic reconstruction of manuscript Tables 6–10.
- `scripts/generate_figures.py`: deterministic reconstruction of manuscript Figures 1–5.
- `scripts/verify_repository.py`: configuration, provenance, result, table and figure audit.

## Intentionally excluded

- Raw CICIDS2017 and CSE-CIC-IDS2018 files.
- Frozen Parquet partitions and source-median imputer binary.
- XGBoost and Random Forest model binaries.
- Per-condition prediction arrays.

These exclusions keep the public package small and avoid dataset redistribution. The frozen protocols identify the excluded authoritative artifacts by size and SHA-256 hash.
