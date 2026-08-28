"""Validate and materialize the frozen revision-v2 experiment specification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocols" / "revision_v2"
PROTOCOL_PATH = PROTOCOL_DIR / "revision_v2_protocol_frozen.json"
PARTITIONS_PATH = PROTOCOL_DIR / "random_partitions_frozen.json"
MATRIX_PATH = PROTOCOL_DIR / "revision_v2_job_matrix.csv"
AUDIT_PATH = PROTOCOL_DIR / "revision_v2_protocol_audit.json"
METADATA_PATH = ROOT / "config" / "experiment_metadata.json"
GROUPS_PATH = ROOT / "config" / "semantic_feature_groups.json"


def canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def expected_random_partitions(protocol: dict, metadata: dict) -> dict:
    predictors = metadata["predictors"]
    sizes = protocol["predictor_schema"]["semantic_group_sizes"]
    seeds = protocol["variants"]["random_partition_group_mask"][
        "partition_seeds"
    ]
    partitions: dict[str, dict[str, list[str]]] = {}
    for seed in seeds:
        permutation = np.random.default_rng(seed).permutation(len(predictors))
        groups: dict[str, list[str]] = {}
        cursor = 0
        for position, size in enumerate(sizes, start=1):
            indices = permutation[cursor : cursor + size]
            groups[f"R{position}"] = [predictors[int(index)] for index in indices]
            cursor += size
        partitions[str(seed)] = groups
    return {
        "generation_algorithm": protocol["variants"][
            "random_partition_group_mask"
        ]["partition_generation"],
        "feature_order_source": "config/experiment_metadata.json/predictors",
        "group_sizes": sizes,
        "partitions": partitions,
    }


def add_job(
    jobs: list[dict[str, object]],
    direction: str,
    variant: str,
    model_seed: int,
    action: str,
    partition_seed: int | None = None,
) -> None:
    partition_token = f"_partition_{partition_seed}" if partition_seed else ""
    job_id = f"{direction}_{variant}{partition_token}_seed_{model_seed}"
    jobs.append(
        {
            "job_id": job_id,
            "direction": direction,
            "variant": variant,
            "partition_seed": "" if partition_seed is None else partition_seed,
            "model_seed": model_seed,
            "action": action,
            "training_replacement": "source_training_median",
            "output_subdir": f"revision_v2/{direction}/{job_id}",
        }
    )


def expected_jobs(protocol: dict) -> list[dict[str, object]]:
    seeds = protocol["model"]["confirmatory_model_seeds"]
    partition_seeds = protocol["variants"]["random_partition_group_mask"][
        "partition_seeds"
    ]
    jobs: list[dict[str, object]] = []

    for seed in seeds:
        add_job(jobs, "forward", "matched_complete_baseline", seed, "train_new")
        add_job(jobs, "forward", "iid_matched_feature_dropout", seed, "train_new")
        for partition_seed in partition_seeds:
            add_job(
                jobs,
                "forward",
                "random_partition_group_mask",
                seed,
                "train_new",
                partition_seed,
            )
        add_job(
            jobs,
            "forward",
            "semantic_uniform_singleton",
            seed,
            "reuse_verified_v1_augmentation_only",
        )
        add_job(
            jobs,
            "forward",
            "semantic_exhaustive_singleton",
            seed,
            "train_new",
        )
        add_job(
            jobs,
            "forward",
            "seen_pairwise_oracle_upper_bound",
            seed,
            "train_new",
        )

    for seed in seeds:
        add_job(jobs, "reverse", "matched_complete_baseline", seed, "train_new")
        add_job(jobs, "reverse", "iid_matched_feature_dropout", seed, "train_new")
        for partition_seed in partition_seeds:
            add_job(
                jobs,
                "reverse",
                "random_partition_group_mask",
                seed,
                "train_new",
                partition_seed,
            )
        add_job(jobs, "reverse", "semantic_uniform_singleton", seed, "train_new")

    return jobs


def matrix_text(jobs: list[dict[str, object]]) -> str:
    columns = [
        "job_id",
        "direction",
        "variant",
        "partition_seed",
        "model_seed",
        "action",
        "training_replacement",
        "output_subdir",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(jobs)
    return buffer.getvalue()


def validate(protocol: dict, metadata: dict, groups_payload: dict) -> list[str]:
    problems: list[str] = []
    check(
        protocol.get("status") == "FROZEN_BEFORE_ANY_REVISION_V2_TRAINING",
        "Protocol status is not frozen-before-training.",
        problems,
    )
    check(protocol.get("protocol_version") == "2.0.0", "Version mismatch.", problems)
    predictor_count = protocol["predictor_schema"]["predictor_count"]
    check(
        len(metadata.get("predictors", [])) == predictor_count,
        "Predictor count mismatch.",
        problems,
    )
    groups = groups_payload.get("semantic_groups", {})
    flattened = [feature for values in groups.values() for feature in values]
    check(
        len(flattened) == predictor_count,
        "Semantic assignment count mismatch.",
        problems,
    )
    check(
        len(flattened) == len(set(flattened)),
        "Duplicate semantic assignments.",
        problems,
    )
    check(
        set(flattened) == set(metadata.get("predictors", [])),
        "Semantic features differ from metadata.",
        problems,
    )
    check(
        [len(values) for values in groups.values()]
        == protocol["predictor_schema"]["semantic_group_sizes"],
        "Semantic group sizes differ from the protocol.",
        problems,
    )
    check(
        protocol["model"]["confirmatory_model_seeds"]
        == [2027, 2028, 2029, 2030, 2031],
        "Confirmatory seed set changed.",
        problems,
    )
    partition_seeds = protocol["variants"]["random_partition_group_mask"][
        "partition_seeds"
    ]
    check(
        len(partition_seeds) == 5,
        "Expected five random partition seeds.",
        problems,
    )
    check(
        len(partition_seeds) == len(set(partition_seeds)),
        "Duplicate partition seeds.",
        problems,
    )
    check(
        protocol["source_only_continuation_gate"]["uses_forward_target_results"]
        is False,
        "Continuation gate must not inspect forward target results.",
        problems,
    )
    check(
        protocol["evaluation"]["condition_rows_per_model_dataset"] == 43,
        "Condition/replacement row count must be 43.",
        problems,
    )
    check(
        protocol["lineage"]["submitted_release_immutable"] is True,
        "Submitted release must remain immutable.",
        problems,
    )
    jobs = expected_jobs(protocol)
    counts = protocol["expected_training_jobs"]
    forward = [job for job in jobs if job["direction"] == "forward"]
    reverse = [job for job in jobs if job["direction"] == "reverse"]
    reused = [job for job in jobs if str(job["action"]).startswith("reuse")]
    check(
        len(forward) == counts["forward_total"],
        "Forward job count mismatch.",
        problems,
    )
    check(
        len(reverse) == counts["reverse_total_if_gate_passes"],
        "Reverse job count mismatch.",
        problems,
    )
    check(
        len(jobs) == counts["grand_total_if_gate_passes"],
        "Grand job count mismatch.",
        problems,
    )
    check(
        len(reused) == counts["grand_total_reused"],
        "Reused job count mismatch.",
        problems,
    )
    check(
        len(jobs) - len(reused) == counts["grand_total_new_if_gate_passes"],
        "New job count mismatch.",
        problems,
    )
    return problems


def build_audit(
    protocol: dict,
    jobs: list[dict[str, object]],
    partition_payload: dict,
    matrix_payload: str,
) -> dict:
    forward = [job for job in jobs if job["direction"] == "forward"]
    reverse = [job for job in jobs if job["direction"] == "reverse"]
    reused = [job for job in jobs if str(job["action"]).startswith("reuse")]
    return {
        "status": "PASS",
        "protocol_version": protocol["protocol_version"],
        "frozen_on_utc": protocol["frozen_on_utc"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "random_partitions_sha256": sha256_bytes(
            canonical_json(partition_payload).encode("utf-8")
        ),
        "job_matrix_sha256": sha256_bytes(matrix_payload.encode("utf-8")),
        "job_counts": {
            "forward": len(forward),
            "reverse_if_gate_passes": len(reverse),
            "total_if_gate_passes": len(jobs),
            "new_if_gate_passes": len(jobs) - len(reused),
            "reused": len(reused),
        },
        "random_partition_count": len(partition_payload["partitions"]),
        "condition_rows_per_model_dataset": protocol["evaluation"][
            "condition_rows_per_model_dataset"
        ],
        "nonselective_rows_if_gate_passes": protocol[
            "expected_evaluation_rows_if_gate_passes"
        ]["nonselective"],
        "selective_rows_if_gate_passes": protocol[
            "expected_evaluation_rows_if_gate_passes"
        ]["selective_at_three_coverages"],
        "problems": [],
    }


def write_or_verify(path: Path, expected: str, materialize: bool) -> None:
    if materialize and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        return
    if not path.exists():
        raise FileNotFoundError(
            f"Missing derived frozen file: {path.relative_to(ROOT)}"
        )
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError(
            f"Frozen derived file differs: {path.relative_to(ROOT)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_json(PROTOCOL_PATH)
    metadata = load_json(METADATA_PATH)
    groups_payload = load_json(GROUPS_PATH)
    problems = validate(protocol, metadata, groups_payload)
    if problems:
        raise RuntimeError(
            "Revision-v2 protocol validation failed: " + "; ".join(problems)
        )

    partitions = expected_random_partitions(protocol, metadata)
    jobs = expected_jobs(protocol)
    matrix_payload = matrix_text(jobs)
    audit = build_audit(protocol, jobs, partitions, matrix_payload)

    write_or_verify(
        PARTITIONS_PATH,
        canonical_json(partitions),
        args.materialize,
    )
    write_or_verify(MATRIX_PATH, matrix_payload, args.materialize)
    write_or_verify(AUDIT_PATH, canonical_json(audit), args.materialize)

    print(canonical_json(audit), end="")


if __name__ == "__main__":
    main()
