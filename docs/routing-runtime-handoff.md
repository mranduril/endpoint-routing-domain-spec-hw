# Routing Runtime Handoff

This project is a prototype runtime for routing logical jobs across
heterogeneous endpoints. The current target story is a two-node system where
each node has one CPU endpoint, one GPU endpoint, and one simulated hardware
endpoint. The simulated endpoint is backed by SimPy traces for performance
modeling, while real numerical work is still performed by CPU fallback code so
examples remain correct out of the box.

## Main Idea

A user submits a logical job, such as SAXPY or one phase of a Jacobi stencil.
The router estimates the cost of running that job on CPU, GPU, CPU/GPU split,
or SIM. It then returns a `DispatchPlan`. The runtime executes the selected
endpoint implementation and writes logs for later analysis.

The design mirrors database query optimization:

```text
logical job        -> logical operator
CPU/GPU/SIM choice -> physical plan
cost model         -> plan estimator
runtime dispatch   -> executor
```

## Core Files

`include/workload_type.h`

Defines the job payloads and metadata:

- `payloadSAXPY` stores scalar/vector information.
- `payloadJacobi` stores grid dimensions, halo width, input, and output.
- `JobMetadata` stores job ID, node ID, iteration, neighbor node ID, and data
  residency hints.
- `DataLocation` is used by the router to decide whether a data movement
  penalty should be charged.

`include/job.h` and `src/job.cpp`

Define the logical `Job` object and factory helpers:

- `make_saxpy(...)`
- `make_jacobi(...)`

`Job::validate()` ensures payload type and buffer sizes are consistent before
the router sees the job.

`include/router.h` and `src/router.cpp`

Define the cost model and routing decisions. `RouterConfig` contains tunable
parameters for queueing, fixed overheads, transfer costs, per-work-unit costs,
and remote-node penalties.

`src/runtime.cpp`

Owns asynchronous request submission, routing-log generation, runtime queue
pressure counters, and endpoint dispatch.

`src/kernels_cpu.cpp`

Contains real CPU work for SAXPY and Jacobi.

`src/kernels_cuda.cu`

Contains GPU SAXPY and GPU Jacobi interior support.

`src/kernels_sim.cpp`

Contains SIM endpoint operators. These operators emit SimPy input traces and
then run the corresponding CPU implementation for correctness.

`prototype/sim_endpoint.py`

Consumes SIM traces and produces simulated timing/output traces. This is the
offline SimPy model for performance/cost verification.

## Workload Decomposition

SAXPY is one logical operation:

```text
SAXPY
```

Jacobi is decomposed into phases:

```text
JACOBI_INTERIOR
JACOBI_BOUNDARY
JACOBI_HALO_BOUNDARY
```

The split exists because different phases have different endpoint preferences:

- Interior work is large regular computation and is GPU-friendly.
- Boundary work is smaller and communication-sensitive.
- Halo-boundary work represents the DPU/SmartNIC/SIM endpoint story.

## Cost Model

The current router estimates:

```text
cost_cpu =
    queue_cpu
  + cpu_fixed
  + cpu_per_work_unit * work_units
  + host_movement_penalty_if_needed

cost_gpu =
    queue_gpu
  + copy_to_gpu_if_needed
  + cuda_launch
  + gpu_per_work_unit * work_units
  + copy_back_if_needed

cost_sim =
    queue_sim
  + modeled_transfer_in_if_needed
  + sim_setup
  + sim_per_work_unit * work_units
  + modeled_transfer_out_if_needed

remote_penalty =
    remote_fixed
  + remote_transfer_per_byte * (input_bytes + output_bytes)
```

The remote penalty is added when `job.metadata.node_id` differs from
`RouterConfig::local_node_id`.

## Data Location

Jobs carry data-location hints:

```cpp
metadata.input_location = DataLocation::Host;
metadata.output_location = DataLocation::Host;
```

Defaults are host-to-host because the current examples use normal C++ vectors.
If data is already on GPU, GPU input-copy cost can be skipped. If output should
remain on GPU, GPU copy-back cost can be skipped. The same pattern exists for
SIM and CPU-side host movement.

This is only a cost-model hint today. The runtime does not yet maintain a real
data-residency table or automatically update data locations after execution.

## Queue Pressure

The runtime keeps lightweight in-flight counters:

```text
g_cpu_jobs
g_gpu_jobs
g_sim_jobs
```

Before planning a new job, the runtime copies these counters into
`RouterConfig`. The router converts them into queue penalties:

```text
effective_queue = base_queue + queued_jobs * queue_job_penalty
```

This gives the router a simple way to avoid endpoints that are already busy.
It is not a precise scheduler or event simulator; it is a cheap pressure signal.

## SIM Endpoint Design

The SIM endpoint does two things:

1. Emit a JSONL trace record under `outputs/sim_traces`.
2. Execute the real math through CPU fallback.

This means example programs remain numerically correct even when forced to SIM.
SimPy is then run offline:

```bash
python -m prototype.run_simpy \
  --input outputs/sim_traces/routing_jacobi_sim_jobs.jsonl \
  --output outputs/sim_traces/routing_jacobi_sim_results.jsonl \
  --summary-output outputs/routing_jacobi_sim_summary.jsonl
```

The SIM trace contains job IDs, node IDs, operation type, arrival time, byte
counts, dimensions, and metadata.

## Output Files

Examples emit:

```text
outputs/sim_traces/routing_saxpy_sim_jobs.jsonl
outputs/sim_traces/routing_jacobi_sim_jobs.jsonl
outputs/routing_saxpy_run_log_<timestamp>.jsonl
outputs/routing_jacobi_run_log_<timestamp>.jsonl
```

Run logs contain routing metadata, chosen endpoint, costs, byte counts, ranges,
target node, data-location hints, and remote cost.

SIM summaries are produced by the Python SimPy runner and describe simulated
busy time, makespan, and utilization.

## Multi-Node Status

Multi-node support is structural, not executable MPI support yet.

Implemented:

- Jobs can target a node through `metadata.node_id`.
- Plans are labeled `Local` or `Remote`.
- Remote jobs get a remote cost penalty.
- Logs expose `target_node_id` and `remote` cost.
- SIM traces include `node_id` and endpoint names like `SIM0`.

Not implemented yet:

- MPI process launch.
- Real halo exchange between nodes.
- Remote endpoint execution.
- Distributed data placement tracking.

The next implementation step should be a two-process Jacobi driver using MPI,
where each rank owns one node's local tile and emits SIM traces for its own SIM
endpoint.

## Validation Commands

Build and run local examples:

```bash
make all
./build/bin/saxpy_small
./build/bin/jacobi
```

Run SimPy traces:

```bash
eval "$(conda shell.zsh hook)"
conda activate drl
python -m unittest prototype.test_sim_endpoint
python -m prototype.run_simpy \
  --input outputs/sim_traces/routing_saxpy_sim_jobs.jsonl \
  --output outputs/sim_traces/routing_saxpy_sim_results.jsonl \
  --summary-output outputs/routing_saxpy_sim_summary.jsonl
python -m prototype.run_simpy \
  --input outputs/sim_traces/routing_jacobi_sim_jobs.jsonl \
  --output outputs/sim_traces/routing_jacobi_sim_results.jsonl \
  --summary-output outputs/routing_jacobi_sim_summary.jsonl
```

## Design Caveats

- SIM execution currently uses CPU fallback for numerical correctness.
- Queue pressure is approximate and based on in-flight request counts.
- Data location is metadata-driven and not automatically tracked.
- Remote-node support affects planning/logging only; actual remote execution is
  not implemented.
- The cost constants are placeholders and should be calibrated from benchmark
  results and SimPy summaries.

