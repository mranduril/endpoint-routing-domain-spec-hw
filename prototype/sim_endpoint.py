"""Offline SimPy model for simulated routing endpoints.

The model represents a SmartNIC/DPU-style stencil accelerator. It consumes
logical jobs emitted by the C++ runtime and produces timing records that can be
used to validate schedules and calibrate the router's SIM cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import simpy


SUPPORTED_OPS = {
    "SAXPY",
    "JACOBI_INTERIOR",
    "JACOBI_BOUNDARY",
    "JACOBI_HALO_BOUNDARY",
}


@dataclass(frozen=True)
class SimEndpointConfig:
    """Timing parameters for one simulated endpoint."""

    setup_us: float = 8.0
    bandwidth_bytes_per_us: float = 12000.0
    saxpy_elements_per_us: float = 8000.0
    jacobi_cells_per_us: float = 4000.0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.setup_us < 0.0:
            raise ValueError("setup_us must be non-negative")
        if self.bandwidth_bytes_per_us <= 0.0:
            raise ValueError("bandwidth_bytes_per_us must be positive")
        if self.saxpy_elements_per_us <= 0.0:
            raise ValueError("saxpy_elements_per_us must be positive")
        if self.jacobi_cells_per_us <= 0.0:
            raise ValueError("jacobi_cells_per_us must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")


@dataclass(frozen=True)
class SimTiming:
    """Breakdown of modeled service time for a job."""

    transfer_in_us: float
    setup_us: float
    compute_us: float
    transfer_out_us: float

    @property
    def service_us(self) -> float:
        return (
            self.transfer_in_us
            + self.setup_us
            + self.compute_us
            + self.transfer_out_us
        )


class SimEndpointModel:
    """SimPy trace consumer for SIM endpoint jobs."""

    def __init__(
        self,
        config: SimEndpointConfig | None = None,
        endpoint_configs: dict[str, SimEndpointConfig] | None = None,
    ) -> None:
        self.default_config = config or SimEndpointConfig()
        self.endpoint_configs = endpoint_configs or {}

    def simulate(self, jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_jobs = sorted(
            (self._normalize_job(job) for job in jobs),
            key=lambda job: (job["arrival_us"], job["job_id"]),
        )

        env = simpy.Environment()
        resources: dict[str, simpy.Resource] = {}
        busy_by_endpoint: dict[str, float] = {}
        results: list[dict[str, Any]] = []

        for job in normalized_jobs:
            endpoint = job["endpoint"]
            config = self._config_for(endpoint)
            if endpoint not in resources:
                resources[endpoint] = simpy.Resource(
                    env,
                    capacity=config.max_concurrency,
                )
                busy_by_endpoint[endpoint] = 0.0
            env.process(
                self._run_job(
                    env,
                    job,
                    resources[endpoint],
                    config,
                    busy_by_endpoint,
                    results,
                )
            )

        env.run()

        windows_by_endpoint = self._active_windows_by_endpoint(results)
        for record in results:
            endpoint = record["endpoint"]
            window_us = windows_by_endpoint[endpoint]
            busy_us = busy_by_endpoint[endpoint]
            record["utilization_window_us"] = window_us
            record["endpoint_busy_us"] = busy_us
            record["endpoint_utilization"] = (
                busy_us / (window_us * self._config_for(endpoint).max_concurrency)
                if window_us > 0.0
                else 0.0
            )

        return sorted(results, key=lambda result: result["job_id"])

    def estimate_timing(self, job: dict[str, Any]) -> SimTiming:
        normalized_job = self._normalize_job(job)
        config = self._config_for(normalized_job["endpoint"])
        cells_or_elements = self._work_units(normalized_job)
        compute_rate = (
            config.saxpy_elements_per_us
            if normalized_job["op"] == "SAXPY"
            else config.jacobi_cells_per_us
        )
        return SimTiming(
            transfer_in_us=normalized_job["input_bytes"]
            / config.bandwidth_bytes_per_us,
            setup_us=config.setup_us,
            compute_us=cells_or_elements / compute_rate,
            transfer_out_us=normalized_job["output_bytes"]
            / config.bandwidth_bytes_per_us,
        )

    def endpoint_summaries(
        self,
        results: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records = list(results)
        endpoints = sorted({record["endpoint"] for record in records})
        summaries = []
        for endpoint in endpoints:
            endpoint_records = [
                record for record in records if record["endpoint"] == endpoint
            ]
            config = self._config_for(endpoint)
            busy_us = sum(record["total_us"] for record in endpoint_records)
            min_arrival_us = min(
                record["arrival_us"] for record in endpoint_records
            )
            max_finish_us = max(record["finish_us"] for record in endpoint_records)
            window_us = max_finish_us - min_arrival_us
            summaries.append(
                {
                    "endpoint": endpoint,
                    "busy_us": busy_us,
                    "makespan_us": window_us,
                    "capacity": config.max_concurrency,
                    "utilization": (
                        busy_us / (window_us * config.max_concurrency)
                        if window_us > 0.0
                        else 0.0
                    ),
                }
            )
        return summaries

    def _run_job(
        self,
        env: simpy.Environment,
        job: dict[str, Any],
        resource: simpy.Resource,
        config: SimEndpointConfig,
        busy_by_endpoint: dict[str, float],
        results: list[dict[str, Any]],
    ) -> Any:
        yield env.timeout(max(0, job["arrival_us"] - env.now))
        with resource.request() as request:
            yield request
            start_us = float(env.now)
            timing = self.estimate_timing(job)
            yield env.timeout(timing.service_us)
            finish_us = float(env.now)

        endpoint = job["endpoint"]
        busy_by_endpoint[endpoint] += timing.service_us
        results.append(
            {
                "job_id": job["job_id"],
                "node_id": job["node_id"],
                "endpoint": endpoint,
                "op": job["op"],
                "arrival_us": job["arrival_us"],
                "start_us": start_us,
                "finish_us": finish_us,
                "queue_us": start_us - job["arrival_us"],
                "latency_us": finish_us - job["arrival_us"],
                "transfer_in_us": timing.transfer_in_us,
                "setup_us": timing.setup_us,
                "compute_us": timing.compute_us,
                "transfer_out_us": timing.transfer_out_us,
                "total_us": timing.service_us,
                "metadata": job.get("metadata", {}),
                "capacity": config.max_concurrency,
            }
        )

    def _config_for(self, endpoint: str) -> SimEndpointConfig:
        return self.endpoint_configs.get(endpoint, self.default_config)

    @staticmethod
    def _active_windows_by_endpoint(
        results: Iterable[dict[str, Any]],
    ) -> dict[str, float]:
        min_arrivals: dict[str, float] = {}
        max_finishes: dict[str, float] = {}
        for record in results:
            endpoint = record["endpoint"]
            min_arrivals[endpoint] = min(
                min_arrivals.get(endpoint, record["arrival_us"]),
                record["arrival_us"],
            )
            max_finishes[endpoint] = max(
                max_finishes.get(endpoint, record["finish_us"]),
                record["finish_us"],
            )
        return {
            endpoint: max_finishes[endpoint] - min_arrivals[endpoint]
            for endpoint in max_finishes
        }

    @staticmethod
    def _normalize_job(job: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(job)
        if "job_id" not in normalized:
            raise ValueError("job is missing required field job_id")

        normalized.setdefault("node_id", 0)
        normalized.setdefault("endpoint", f"SIM{normalized['node_id']}")
        normalized.setdefault("arrival_us", 0.0)
        normalized.setdefault("input_bytes", 0.0)
        normalized.setdefault("output_bytes", 0.0)
        normalized.setdefault("metadata", {})

        op = normalized.get("op")
        if op not in SUPPORTED_OPS:
            raise ValueError(f"unsupported op: {op}")

        normalized["arrival_us"] = float(normalized["arrival_us"])
        normalized["input_bytes"] = float(normalized["input_bytes"])
        normalized["output_bytes"] = float(normalized["output_bytes"])
        if normalized["arrival_us"] < 0.0:
            raise ValueError("arrival_us must be non-negative")
        if normalized["input_bytes"] < 0.0 or normalized["output_bytes"] < 0.0:
            raise ValueError("input_bytes/output_bytes must be non-negative")

        return normalized

    @staticmethod
    def _work_units(job: dict[str, Any]) -> float:
        if job["op"] == "SAXPY":
            n = float(job["n"])
            if n < 0.0:
                raise ValueError("n must be non-negative")
            return n

        nx = int(job["nx"])
        ny = int(job["ny"])
        halo_width = int(job.get("halo_width", 1))
        if nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive")
        if halo_width <= 0:
            raise ValueError("halo_width must be positive")

        if job["op"] == "JACOBI_INTERIOR":
            interior_nx = max(nx - 2 * halo_width, 0)
            interior_ny = max(ny - 2 * halo_width, 0)
            return float(interior_nx * interior_ny)

        if "boundary_cells" in job:
            boundary_cells = float(job["boundary_cells"])
            if boundary_cells < 0.0:
                raise ValueError("boundary_cells must be non-negative")
            return boundary_cells

        horizontal = min(2 * halo_width, ny) * nx
        vertical = max(ny - 2 * halo_width, 0) * min(2 * halo_width, nx)
        return float(horizontal + vertical)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True))
            output_file.write("\n")
