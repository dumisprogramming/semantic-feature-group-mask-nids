"""Prepare the frozen CICIDS2017 -> CSE-CIC-IDS2018 experiment partitions.

This script is a cleaned, command-line version of the data-preparation cells in
``notebooks/authoritative_experiment_notebook.ipynb``.  Sampling is uniform and
independent of the binary label.  Target labels are written to the sealed test
partition but are not used for training, calibration, threshold selection, or
model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "semantic_feature_groups.json"

MAPPING_2018_TO_2017 = {
    "ACK Flag Cnt": "ACK Flag Count",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Header Len": "Bwd Header Length",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Dst Port": "Destination Port",
    "ECE Flag Cnt": "ECE Flag Count",
    "FIN Flag Cnt": "FIN Flag Count",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Header Len": "Fwd Header Length",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Fwd Seg Size Min": "min_seg_size_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "PSH Flag Cnt": "PSH Flag Count",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Pkt Size Avg": "Average Packet Size",
    "RST Flag Cnt": "RST Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "URG Flag Cnt": "URG Flag Count",
}

DROP_2018 = ["Protocol", "Timestamp", "Flow ID", "Src IP", "Dst IP", "Src Port"]
INVALID_LABELS = {"label", "", "nan"}


def load_semantic_configuration(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    groups = payload["semantic_groups"]
    predictors = sorted(feature for values in groups.values() for feature in values)
    if len(predictors) != payload["predictor_count"]:
        raise ValueError("Semantic-group predictor count does not match the configuration.")
    if len(predictors) != len(set(predictors)):
        raise ValueError("A predictor is assigned to more than one semantic group.")
    return groups, predictors


def standardize_chunk(chunk: pd.DataFrame, dataset: str) -> pd.DataFrame:
    chunk.columns = [str(column).strip() for column in chunk.columns]
    if dataset == "2017":
        return chunk.drop(columns=["Fwd Header Length.1"], errors="ignore")
    if dataset == "2018":
        return chunk.drop(columns=DROP_2018, errors="ignore").rename(
            columns=MAPPING_2018_TO_2017
        )
    raise ValueError(f"Unsupported dataset identifier: {dataset}")


def normalized_header(path: Path, dataset: str) -> set[str]:
    frame = pd.read_csv(path, nrows=0)
    frame = standardize_chunk(frame, dataset)
    return set(frame.columns)


def validate_schemas(
    source_files: list[Path], target_files: list[Path], predictors: list[str]
) -> None:
    expected = set(predictors) | {"Label"}
    problems: list[str] = []
    for dataset, files in (("2017", source_files), ("2018", target_files)):
        for path in files:
            header = normalized_header(path, dataset)
            if header != expected:
                problems.append(
                    f"{dataset}/{path.name}: missing={sorted(expected-header)}; "
                    f"additional={sorted(header-expected)}"
                )
    if problems:
        raise ValueError("Schema validation failed:\n" + "\n".join(problems))


def clean_chunk(
    raw_chunk: pd.DataFrame, dataset: str, predictors: list[str]
) -> pd.DataFrame:
    chunk = standardize_chunk(raw_chunk, dataset)
    labels = chunk["Label"].astype(str).str.strip()
    valid = ~labels.str.lower().isin(INVALID_LABELS)
    chunk = chunk.loc[valid].copy()
    labels = labels.loc[valid]
    chunk["target"] = (labels.str.lower() != "benign").astype("int8")
    chunk[predictors] = chunk[predictors].apply(pd.to_numeric, errors="coerce")
    chunk[predictors] = chunk[predictors].replace([np.inf, -np.inf], np.nan)
    return chunk[predictors + ["target"]]


def create_uniform_sample(
    files: list[Path],
    dataset: str,
    predictors: list[str],
    valid_population: int,
    desired_rows: int,
    seed: int,
    chunksize: int,
) -> pd.DataFrame:
    probability = min(1.0, desired_rows / valid_population)
    rng = np.random.default_rng(seed)
    samples: list[pd.DataFrame] = []
    print(f"Sampling {dataset}; Bernoulli probability={probability:.10f}")
    for number, path in enumerate(files, start=1):
        start = time.time()
        selected_count = 0
        for raw_chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            encoding_errors="replace",
        ):
            cleaned = clean_chunk(raw_chunk, dataset, predictors)
            selected = cleaned.loc[rng.random(len(cleaned)) < probability].copy()
            selected[predictors] = selected[predictors].astype("float32")
            selected["target"] = selected["target"].astype("int8")
            samples.append(selected)
            selected_count += len(selected)
        print(
            f"[{number}/{len(files)}] {path.name}: selected={selected_count:,}; "
            f"seconds={time.time()-start:.1f}"
        )
    if not samples:
        raise RuntimeError(f"No rows were sampled from {dataset}.")
    sample = pd.concat(samples, ignore_index=True)
    print(
        f"{dataset}: rows={len(sample):,}; attacks={int(sample['target'].sum()):,}; "
        f"attack_rate={sample['target'].mean():.6f}"
    )
    return sample


def hash_partition(source: pd.DataFrame, predictors: list[str]) -> dict[str, pd.DataFrame]:
    feature_hash = pd.util.hash_pandas_object(source[predictors], index=False).to_numpy(
        dtype="uint64"
    )
    bucket = feature_hash % 100
    return {
        "source_train": source.loc[bucket < 60].copy(),
        "source_validation": source.loc[(bucket >= 60) & (bucket < 75)].copy(),
        "source_calibration": source.loc[(bucket >= 75) & (bucket < 85)].copy(),
        "source_test": source.loc[bucket >= 85].copy(),
    }


def save_outputs(
    partitions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    predictors: list[str],
    semantic_groups: dict[str, list[str]],
    output_dir: Path,
    source_seed: int,
    target_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(partitions["source_train"][predictors])
    for name, frame in partitions.items():
        frame.to_parquet(output_dir / f"{name}.parquet", compression="zstd", index=False)
    target.to_parquet(
        output_dir / "target_test_sealed.parquet", compression="zstd", index=False
    )
    joblib.dump(imputer, output_dir / "source_median_imputer.joblib")
    metadata = {
        "source_dataset": "CICIDS2017",
        "target_dataset": "CSE-CIC-IDS2018",
        "source_sampling_seed": source_seed,
        "target_sampling_seed": target_seed,
        "predictor_count": len(predictors),
        "predictors": predictors,
        "semantic_groups": semantic_groups,
        "partition_rows": {
            "training": len(partitions["source_train"]),
            "validation": len(partitions["source_validation"]),
            "calibration": len(partitions["source_calibration"]),
            "source_test": len(partitions["source_test"]),
            "target_test": len(target),
        },
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print("Saved frozen partitions and source-fitted imputer to", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cicids2017-dir", type=Path, required=True)
    parser.add_argument("--cse-cic-ids2018-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-population", type=int, default=2_830_743)
    parser.add_argument("--target-population", type=int, default=16_232_943)
    parser.add_argument("--sample-rows", type=int, default=1_000_000)
    parser.add_argument("--source-seed", type=int, default=2026)
    parser.add_argument("--target-seed", type=int, default=2027)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--schema-only", action="store_true", help="Validate files without sampling."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups, predictors = load_semantic_configuration(args.config)
    source_files = sorted(args.cicids2017_dir.rglob("*.csv"))
    target_files = sorted(
        path
        for path in args.cse_cic_ids2018_dir.rglob("*.csv")
        if "TrafficForML" in path.name
    )
    if not source_files or not target_files:
        raise FileNotFoundError(
            f"Found {len(source_files)} source and {len(target_files)} target CSV files."
        )
    print(f"Files: CICIDS2017={len(source_files)}; CSE-CIC-IDS2018={len(target_files)}")
    validate_schemas(source_files, target_files, predictors)
    print("Schema validation: PASS")
    if args.schema_only:
        return
    source = create_uniform_sample(
        source_files,
        "2017",
        predictors,
        args.source_population,
        args.sample_rows,
        args.source_seed,
        args.chunksize,
    )
    target = create_uniform_sample(
        target_files,
        "2018",
        predictors,
        args.target_population,
        args.sample_rows,
        args.target_seed,
        args.chunksize,
    )
    partitions = hash_partition(source, predictors)
    save_outputs(
        partitions,
        target,
        predictors,
        groups,
        args.output_dir,
        args.source_seed,
        args.target_seed,
    )


if __name__ == "__main__":
    main()
