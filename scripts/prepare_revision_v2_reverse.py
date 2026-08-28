"""Prepare and freeze CSE-CIC-IDS2018 -> CICIDS2017 revision-v2 artifacts.

This command is stage-gated by the frozen forward source-only continuation
decision.  It samples CSE-CIC-IDS2018 as the reverse source, partitions that
source with the same deterministic feature-hash rule as v1, fits a median
imputer on the reverse source-training partition only, writes a sealed
CICIDS2017 target, and creates a SHA-256 artifact manifest before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from prepare_data import (
    create_uniform_sample,
    hash_partition,
    load_semantic_configuration,
    normalized_header,
)
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "semantic_feature_groups.json"
DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get(
        "NIDS_CHECKPOINT_DIR",
        "/content/drive/MyDrive/research/researchdata/"
        "NIDS_Research/NIDS_Research_Checkpoints",
    )
)
PROTOCOL_PATH = (
    ROOT / "protocols" / "revision_v2" / ("revision_v2_protocol_frozen.json")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def validate_direction_schemas(
    source_files: list[Path],
    target_files: list[Path],
    predictors: list[str],
) -> None:
    expected = set(predictors) | {"Label"}
    problems = []
    for dataset, files in [("2018", source_files), ("2017", target_files)]:
        for path in files:
            header = normalized_header(path, dataset)
            if header != expected:
                problems.append(
                    f"{dataset}/{path.name}: missing={sorted(expected - header)}; "
                    f"additional={sorted(header - expected)}"
                )
    if problems:
        raise RuntimeError("Reverse schema validation failed:\n" + "\n".join(problems))


def require_forward_gate(checkpoint_dir: Path) -> dict:
    gate_path = (
        checkpoint_dir
        / "revision_v2"
        / "forward"
        / ("FORWARD_SOURCE_ONLY_CONTINUATION_GATE.json")
    )
    if not gate_path.exists():
        raise FileNotFoundError(
            "The frozen forward source-only continuation gate is missing."
        )
    gate = load_json(gate_path)
    if gate.get("status") != "PASS" or not gate.get("reverse_training_authorized"):
        raise RuntimeError("Reverse preparation is blocked by the forward gate.")
    return gate


def save_reverse_outputs(
    partitions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    predictors: list[str],
    semantic_groups: dict[str, list[str]],
    output_dir: Path,
    source_seed: int,
    target_seed: int,
    gate_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(partitions["source_train"][predictors])
    files: list[Path] = []
    for name, frame in partitions.items():
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, compression="zstd", index=False)
        files.append(path)
    target_path = output_dir / "target_test_sealed.parquet"
    target.to_parquet(target_path, compression="zstd", index=False)
    files.append(target_path)
    imputer_path = output_dir / "source_median_imputer.joblib"
    joblib.dump(imputer, imputer_path)
    files.append(imputer_path)
    metadata = {
        "direction": "reverse",
        "source_dataset": "CSE-CIC-IDS2018",
        "target_dataset": "CICIDS2017",
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
        "partition_method": "feature hash modulo 100: 60/15/10/15",
        "imputer_fit_partition": "reverse source training only",
        "target_role": "sealed evaluation only",
    }
    metadata_path = output_dir / "experiment_metadata.json"
    atomic_json(metadata, metadata_path)
    files.append(metadata_path)
    manifest = {
        "status": "reverse_artifacts_frozen_before_training",
        "created_utc": utc_now(),
        "direction": "CSE-CIC-IDS2018_to_CICIDS2017",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "forward_gate_sha256": sha256_file(gate_path),
        "source_labels_used_for": "partitioning, training and source-only development",
        "target_labels_used_for": (
            "read only to persist the sealed target; not summarized, inspected, "
            "or used for training, calibration, policy selection or model selection"
        ),
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    atomic_json(
        manifest,
        output_dir / "REVISION_V2_REVERSE_ARTIFACT_MANIFEST.json",
    )
    print("Reverse artifacts frozen at", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cicids2017-dir", type=Path, required=True)
    parser.add_argument("--cse-cic-ids2018-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--schema-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = require_forward_gate(args.checkpoint_dir)
    protocol = load_json(PROTOCOL_PATH)
    reverse = protocol["directions"]["reverse"]
    groups, predictors = load_semantic_configuration(args.config)
    source_files = sorted(
        path
        for path in args.cse_cic_ids2018_dir.rglob("*.csv")
        if "TrafficForML" in path.name
    )
    target_files = sorted(args.cicids2017_dir.rglob("*.csv"))
    if not source_files or not target_files:
        raise FileNotFoundError(
            f"Found {len(source_files)} reverse-source and "
            f"{len(target_files)} reverse-target CSV files."
        )
    validate_direction_schemas(source_files, target_files, predictors)
    print("Reverse schema validation: PASS")
    if args.schema_only:
        return
    source = create_uniform_sample(
        source_files,
        "2018",
        predictors,
        reverse["source_population_rows"],
        reverse["source_sample_rows"],
        reverse["source_sampling_seed"],
        args.chunksize,
    )
    target = create_uniform_sample(
        target_files,
        "2017",
        predictors,
        reverse["target_population_rows"],
        reverse["target_sample_rows"],
        reverse["target_sampling_seed"],
        args.chunksize,
        report_label_summary=False,
    )
    partitions = hash_partition(source, predictors)
    output_dir = args.checkpoint_dir / "revision_v2" / "reverse" / "artifacts"
    gate_path = (
        args.checkpoint_dir
        / "revision_v2"
        / "forward"
        / ("FORWARD_SOURCE_ONLY_CONTINUATION_GATE.json")
    )
    save_reverse_outputs(
        partitions,
        target,
        predictors,
        groups,
        output_dir,
        reverse["source_sampling_seed"],
        reverse["target_sampling_seed"],
        gate_path,
    )
    print("Forward gate status used:", gate["status"])


if __name__ == "__main__":
    main()
