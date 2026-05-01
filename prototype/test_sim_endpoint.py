import math
import tempfile
import unittest
from pathlib import Path

from prototype.sim_endpoint import (
    SimEndpointConfig,
    SimEndpointModel,
    read_jsonl,
    write_jsonl,
)


class SimEndpointModelTest(unittest.TestCase):
    def setUp(self):
        self.config = SimEndpointConfig(
            setup_us=10.0,
            bandwidth_bytes_per_us=1_000.0,
            saxpy_elements_per_us=100.0,
            jacobi_cells_per_us=50.0,
            max_concurrency=1,
        )
        self.model = SimEndpointModel(config=self.config)

    def test_preserves_jobs_and_causality(self):
        jobs = [
            {
                "job_id": 2,
                "node_id": 0,
                "endpoint": "SIM0",
                "op": "JACOBI_BOUNDARY",
                "arrival_us": 5.0,
                "nx": 10,
                "ny": 8,
                "halo_width": 1,
                "input_bytes": 1000,
                "output_bytes": 500,
            },
            {
                "job_id": 1,
                "node_id": 0,
                "endpoint": "SIM0",
                "op": "JACOBI_BOUNDARY",
                "arrival_us": 0.0,
                "nx": 10,
                "ny": 8,
                "halo_width": 1,
                "input_bytes": 1000,
                "output_bytes": 500,
            },
        ]

        results = self.model.simulate(jobs)

        self.assertEqual([1, 2], [record["job_id"] for record in results])
        for record in results:
            self.assertGreaterEqual(record["start_us"], record["arrival_us"])
            self.assertGreaterEqual(record["finish_us"], record["start_us"])

    def test_capacity_one_jobs_do_not_overlap_on_same_endpoint(self):
        jobs = [
            {
                "job_id": 1,
                "endpoint": "SIM0",
                "op": "SAXPY",
                "arrival_us": 0.0,
                "n": 1000,
            },
            {
                "job_id": 2,
                "endpoint": "SIM0",
                "op": "SAXPY",
                "arrival_us": 1.0,
                "n": 1000,
            },
        ]

        first, second = self.model.simulate(jobs)

        self.assertGreaterEqual(second["start_us"], first["finish_us"])
        self.assertGreater(second["queue_us"], 0.0)

    def test_service_time_matches_breakdown(self):
        [record] = self.model.simulate(
            [
                {
                    "job_id": 1,
                    "endpoint": "SIM0",
                    "op": "JACOBI_HALO_BOUNDARY",
                    "arrival_us": 0.0,
                    "nx": 10,
                    "ny": 8,
                    "halo_width": 1,
                    "input_bytes": 1000,
                    "output_bytes": 500,
                }
            ]
        )

        expected_total = (
            record["transfer_in_us"]
            + record["setup_us"]
            + record["compute_us"]
            + record["transfer_out_us"]
        )
        self.assertTrue(math.isclose(record["total_us"], expected_total))
        self.assertTrue(
            math.isclose(record["finish_us"] - record["start_us"], expected_total)
        )

    def test_distinct_endpoints_can_run_independently(self):
        jobs = [
            {
                "job_id": 1,
                "node_id": 0,
                "endpoint": "SIM0",
                "op": "SAXPY",
                "arrival_us": 0.0,
                "n": 1000,
            },
            {
                "job_id": 2,
                "node_id": 1,
                "endpoint": "SIM1",
                "op": "SAXPY",
                "arrival_us": 0.0,
                "n": 1000,
            },
        ]

        first, second = self.model.simulate(jobs)

        self.assertEqual(first["start_us"], second["start_us"])
        self.assertEqual(first["finish_us"], second["finish_us"])

    def test_jsonl_round_trip(self):
        jobs = [
            {
                "job_id": 1,
                "endpoint": "SIM0",
                "op": "SAXPY",
                "arrival_us": 0.0,
                "n": 100,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "jobs.jsonl"
            output_path = Path(temp_dir) / "results.jsonl"
            write_jsonl(input_path, jobs)
            results = self.model.simulate(read_jsonl(input_path))
            write_jsonl(output_path, results)

            self.assertEqual([1], [record["job_id"] for record in read_jsonl(output_path)])

    def test_utilization_window_uses_first_arrival_not_zero(self):
        jobs = [
            {
                "job_id": 1,
                "endpoint": "SIM0",
                "op": "SAXPY",
                "arrival_us": 100.0,
                "n": 100,
            }
        ]

        [record] = self.model.simulate(jobs)
        [summary] = self.model.endpoint_summaries([record])

        self.assertTrue(
            math.isclose(record["utilization_window_us"], record["latency_us"])
        )
        self.assertTrue(math.isclose(summary["makespan_us"], record["latency_us"]))
        self.assertTrue(math.isclose(record["endpoint_utilization"], 1.0))
        self.assertTrue(math.isclose(summary["utilization"], 1.0))

    def test_rejects_negative_work_units(self):
        with self.assertRaises(ValueError):
            self.model.simulate(
                [
                    {
                        "job_id": 1,
                        "endpoint": "SIM0",
                        "op": "SAXPY",
                        "n": -1,
                    }
                ]
            )

        with self.assertRaises(ValueError):
            self.model.simulate(
                [
                    {
                        "job_id": 2,
                        "endpoint": "SIM0",
                        "op": "JACOBI_BOUNDARY",
                        "nx": 10,
                        "ny": 10,
                        "boundary_cells": -1,
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
