"""Run the offline SimPy endpoint model over a JSONL trace."""

from __future__ import annotations

import argparse

from prototype.sim_endpoint import (
    SimEndpointConfig,
    SimEndpointModel,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate SIM endpoint jobs from a JSONL trace.",
    )
    parser.add_argument("--input", required=True, help="Input job trace JSONL")
    parser.add_argument("--output", required=True, help="Output timing JSONL")
    parser.add_argument("--setup-us", type=float, default=8.0)
    parser.add_argument("--bandwidth-bytes-per-us", type=float, default=12_000.0)
    parser.add_argument("--saxpy-elements-per-us", type=float, default=8_000.0)
    parser.add_argument("--jacobi-cells-per-us", type=float, default=4_000.0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--summary-output",
        help="Optional endpoint utilization summary JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimEndpointConfig(
        setup_us=args.setup_us,
        bandwidth_bytes_per_us=args.bandwidth_bytes_per_us,
        saxpy_elements_per_us=args.saxpy_elements_per_us,
        jacobi_cells_per_us=args.jacobi_cells_per_us,
        max_concurrency=args.max_concurrency,
    )
    model = SimEndpointModel(config=config)
    results = model.simulate(read_jsonl(args.input))
    write_jsonl(args.output, results)
    if args.summary_output:
        write_jsonl(args.summary_output, model.endpoint_summaries(results))


if __name__ == "__main__":
    main()

