"""Dependency-light tests for revision-v2 protocol and runner invariants."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_revision_v2 as runner


def synthetic_definition() -> dict:
    predictors = [f"f{number:02d}" for number in range(77)]
    sizes = [32, 23, 12, 10]
    names = ["G1", "G2", "G3", "G4"]
    groups = {}
    group_indices = {}
    cursor = 0
    for name, size in zip(names, sizes):
        groups[name] = predictors[cursor : cursor + size]
        group_indices[name] = list(range(cursor, cursor + size))
        cursor += size
    return {
        "predictors": predictors,
        "group_names": names,
        "groups": groups,
        "group_indices": group_indices,
        "medians": np.arange(77, dtype=np.float32) + 100,
    }


def synthetic_partitions(definition: dict) -> dict:
    return {
        "partitions": {
            "4101": {
                "R1": definition["predictors"][:32],
                "R2": definition["predictors"][32:55],
                "R3": definition["predictors"][55:67],
                "R4": definition["predictors"][67:],
            }
        }
    }


class ProtocolTests(unittest.TestCase):
    def test_frozen_bundle_counts(self) -> None:
        protocol, audit, partitions, jobs = runner.load_protocol_bundle()
        self.assertEqual(protocol["protocol_version"], "2.0.0")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(partitions["partitions"]), 5)
        self.assertEqual(len(jobs), 90)
        self.assertEqual(len(runner.jobs_for_direction(jobs, "forward")), 50)
        self.assertEqual(len(runner.jobs_for_direction(jobs, "reverse")), 40)

    def test_evaluation_condition_count_and_uniqueness(self) -> None:
        conditions = runner.evaluation_conditions(["G1", "G2", "G3", "G4"])
        self.assertEqual(len(conditions), 43)
        keys = {
            (
                row["severity"],
                row["missing_groups"],
                row["replacement"],
            )
            for row in conditions
        }
        self.assertEqual(len(keys), 43)
        self.assertEqual(sum(row["severity"] == 0 for row in conditions), 1)
        self.assertEqual(sum(row["severity"] == 1 for row in conditions), 12)
        self.assertEqual(sum(row["severity"] == 2 for row in conditions), 18)
        self.assertEqual(sum(row["severity"] == 3 for row in conditions), 12)


class MaskBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition = synthetic_definition()
        self.partitions = synthetic_partitions(self.definition)
        self.x = np.zeros((12, 77), dtype=np.float32)
        self.y = np.asarray([0, 1] * 6, dtype=np.int8)

    def job(self, variant: str, partition_seed=None) -> dict:
        return {
            "variant": variant,
            "model_seed": 2027,
            "partition_seed": partition_seed,
        }

    def test_training_row_multipliers(self) -> None:
        expected = {
            "matched_complete_baseline": 1,
            "iid_matched_feature_dropout": 2,
            "random_partition_group_mask": 2,
            "semantic_uniform_singleton": 2,
            "semantic_exhaustive_singleton": 5,
            "seen_pairwise_oracle_upper_bound": 2,
        }
        for variant, multiplier in expected.items():
            with self.subTest(variant=variant):
                partition_seed = (
                    4101 if variant == "random_partition_group_mask" else None
                )
                x_out, y_out, _ = runner.build_training_data(
                    self.x,
                    self.y,
                    self.job(variant, partition_seed),
                    self.definition,
                    self.partitions,
                )
                self.assertEqual(x_out.shape, (len(self.x) * multiplier, 77))
                self.assertEqual(len(y_out), len(self.y) * multiplier)

    def test_iid_budget_is_exact_per_masked_row(self) -> None:
        masked, counts = runner.iid_masked_copy(
            self.x,
            seed=2027,
            budgets=[32, 23, 12, 10],
            medians=self.definition["medians"],
            chunk_rows=5,
        )
        changed = (masked != 0).sum(axis=1)
        self.assertTrue(set(changed).issubset({32, 23, 12, 10}))
        self.assertEqual(sum(counts.values()), len(masked))

    def test_matched_development_has_five_conditions(self) -> None:
        x_known, y_known = runner.matched_known_conditions(
            self.x, self.y, self.definition
        )
        self.assertEqual(x_known.shape, (len(self.x) * 5, 77))
        self.assertEqual(len(y_known), len(self.y) * 5)


def gate_frame(semantic_pairwise: float, iid_pairwise: float) -> pd.DataFrame:
    rows = []
    for seed in [2027, 2028, 2029, 2030, 2031]:
        for variant, value in [
            ("semantic_uniform_singleton", semantic_pairwise),
            ("iid_matched_feature_dropout", iid_pairwise),
        ]:
            for condition in range(6):
                rows.append(
                    {
                        "direction": "forward",
                        "dataset_role": "source",
                        "severity": 2,
                        "replacement": "source_training_median",
                        "model_seed": seed,
                        "variant": variant,
                        "partition_seed": np.nan,
                        "f1": value + (seed - 2029) * 0.0002,
                    }
                )
        for partition_seed in [4101, 4102, 4103, 4104, 4105]:
            for condition in range(6):
                rows.append(
                    {
                        "direction": "forward",
                        "dataset_role": "source",
                        "severity": 2,
                        "replacement": "source_training_median",
                        "model_seed": seed,
                        "variant": "random_partition_group_mask",
                        "partition_seed": partition_seed,
                        "f1": iid_pairwise - 0.01 + (partition_seed - 4103) * 0.0001,
                    }
                )
        rows.extend(
            [
                {
                    "direction": "forward",
                    "dataset_role": "source",
                    "severity": 0,
                    "replacement": "none",
                    "model_seed": seed,
                    "variant": "semantic_uniform_singleton",
                    "partition_seed": np.nan,
                    "f1": 0.949,
                },
                {
                    "direction": "forward",
                    "dataset_role": "source",
                    "severity": 0,
                    "replacement": "none",
                    "model_seed": seed,
                    "variant": "matched_complete_baseline",
                    "partition_seed": np.nan,
                    "f1": 0.950,
                },
            ]
        )
    return pd.DataFrame(rows)


class GateTests(unittest.TestCase):
    def test_gate_pass(self) -> None:
        result = runner.calculate_forward_gate(gate_frame(0.90, 0.80))
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["reverse_training_authorized"])
        self.assertFalse(result["forward_target_results_used"])

    def test_gate_fail(self) -> None:
        result = runner.calculate_forward_gate(gate_frame(0.75, 0.80))
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["reverse_training_authorized"])


@unittest.skipIf(runner.xgb is None, "XGBoost is not installed")
class EndToEndSmokeTests(unittest.TestCase):
    def test_train_save_reload_and_evaluate(self) -> None:
        rng = np.random.default_rng(123)
        definition = synthetic_definition()
        definition["medians"] = np.zeros(77, dtype=np.float32)

        def sample(rows: int) -> tuple[np.ndarray, np.ndarray]:
            y_data = np.asarray([0, 1] * (rows // 2), dtype=np.int8)
            x_data = rng.normal(size=(rows, 77)).astype(np.float32)
            x_data[:, 0] += y_data * 2.5
            return x_data, y_data

        train = sample(80)
        validation = sample(30)
        calibrator = sample(30)
        policy = sample(30)
        source = {
            "definition": definition,
            "train": train,
            "validation_known": runner.matched_known_conditions(
                *validation, definition
            ),
            "calibrator_known": runner.matched_known_conditions(
                *calibrator, definition
            ),
            "policy_known": runner.matched_known_conditions(*policy, definition),
            "calibration_split": {
                "method": "synthetic",
                "seed": 2026,
                "calibrator_rows": 30,
                "policy_rows": 30,
            },
        }
        protocol, _, partitions, _ = runner.load_protocol_bundle()
        protocol = copy.deepcopy(protocol)
        configuration = protocol["model"]["fixed_configuration"]
        configuration.update(
            {
                "n_estimators": 20,
                "early_stopping_rounds": 5,
                "max_depth": 2,
                "n_jobs": 1,
            }
        )
        job = {
            "job_id": "forward_synthetic_smoke_seed_2027",
            "direction": "forward",
            "variant": "matched_complete_baseline",
            "partition_seed": None,
            "model_seed": 2027,
            "action": "train_new",
            "output_subdir": "revision_v2/forward/forward_synthetic_smoke_seed_2027",
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            runner.train_one_job(
                checkpoint,
                job,
                source,
                protocol,
                partitions,
                "cpu",
            )
            output = runner.job_dir(checkpoint, job)
            self.assertTrue((output / "SOURCE_POLICY_FROZEN.json").exists())
            x_test, y_test = sample(20)
            runner.evaluate_one_job(
                checkpoint,
                job,
                "source",
                definition,
                x_test,
                y_test,
            )
            nonselective = pd.read_csv(output / "source_nonselective.csv")
            selective = pd.read_csv(output / "source_selective.csv")
            self.assertEqual(len(nonselective), 43)
            self.assertEqual(len(selective), 129)
            self.assertTrue((output / "SOURCE_EVALUATION_COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
