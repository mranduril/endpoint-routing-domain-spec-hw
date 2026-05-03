# Technical Design Doc: Cost-Based Endpoint Routing Runtime with SimPy-Modeled Accelerator

## 1. Project Summary

This project prototypes a lightweight heterogeneous computation runtime that routes logical jobs across real and simulated execution endpoints. The current implementation target is constrained to one physical GPU and potentially limited cluster access, so the design uses:

- Real CPU execution
- Real GPU execution through CUDA
- A simulated domain-specific accelerator modeled with SimPy
- Optional two-rank MPI execution for a Jacobi stencil proof of concept

The core idea is to treat CPU, GPU, and simulated hardware as interchangeable physical endpoints for the same logical job. A cost model estimates execution, transfer, setup, and queueing costs, then selects the best endpoint. The system emits traces from the C++ runtime, replays simulated endpoint jobs in Python/SimPy, and produces final evaluation traces and plots.

This project is inspired by compiler/runtime optimization and query optimization: a logical job is analogous to a logical operator, and CPU/GPU/SIM implementations are physical execution plans selected by a cost model.

---

## 2. Goals and Non-Goals

### Goals

1. Provide a unified job/endpoint API for heterogeneous computation.
2. Implement CPU and GPU execution paths for SAXPY and Jacobi-related kernels.
3. Implement a SimPy-based simulated accelerator endpoint that consumes job traces and emits timing/resource traces.
4. Implement a simple cost-model-based router that chooses between CPU, GPU, and SIM endpoints.
5. Evaluate routing quality using:
   - SAXPY on one node
   - Jacobi stencil with two MPI ranks or two logical partitions
6. Produce trace-driven evidence of:
   - endpoint choice
   - cost estimates
   - measured CPU/GPU costs
   - simulated SIM costs
   - routing accuracy
   - utilization and queueing behavior

### Non-Goals

The MVP does **not** aim to:

- Integrate real FPGA/Vortex hardware
- Implement RDMA/RoCE/BlueField/DOCA datapaths
- Build a full compiler frontend or MLIR integration
- Prove true two-node GPU scaling
- Implement a production heterogeneous runtime comparable to StarPU/Legion
- Use SimPy wall-clock time as a benchmark metric

The SimPy endpoint is a performance model, not a real-time executor.

---

## 3. High-Level Architecture

```text
C++ Benchmark / Runtime
  |
  |  emits raw execution trace
  v
raw_trace.jsonl
  |
  |  consumed by
  v
Python SimPy Replay
  |
  |  emits simulated SIM timing/resource trace
  v
sim_trace.jsonl
  |
  |  merged with CPU/GPU measured data
  v
final_trace.jsonl
  |
  v
Evaluation scripts / plots
```

### Major Components

1. **C++ Runtime Layer**
   - Defines `Job`, `Endpoint`, `Router`, `CostModel`, and executors.
   - Runs real CPU/GPU implementations.
   - Emits raw trace records.

2. **CPU Executor**
   - Runs CPU/OpenMP implementations.
   - Measures execution time using `std::chrono`.

3. **GPU Executor**
   - Runs CUDA kernel wrappers.
   - Measures execution time using CUDA events.

4. **SIM Executor / Trace Emitter**
   - Does not run actual hardware.
   - Emits jobs intended for the simulated endpoint.
   - May optionally compute actual output using a CPU reference path for correctness.

5. **Python SimPy Model**
   - Reads trace records for SIM jobs.
   - Models queueing, transfer time, setup latency, compute time, and utilization.
   - Emits simulated timing records.

6. **Evaluation Scripts**
   - Merge C++ raw traces and SimPy traces.
   - Compute routing accuracy, prediction error, endpoint utilization, and policy comparison.
   - Generate plots.

---

## 4. Conceptual Model

### Job

A job is a logical unit of work. It describes **what** should be computed, not **how** or **where**.

Examples:

```text
SAXPY(a, x, y, n)
JACOBI_INTERIOR(tile)
JACOBI_BOUNDARY(tile)
```

A job should include:

- `job_id`
- `workload`
- `op`
- problem size, such as `n`, `nx`, `ny`
- input/output byte sizes
- arrival time
- metadata
- router estimates
- router decision

### Endpoint

An endpoint is a physical or modeled execution target.

Endpoint types:

```text
CPU
GPU
SIM
```

Important: CUDA, MPI, NCCL, and SimPy are **not** endpoints. They are mechanisms used by endpoint executors or simulation infrastructure.

### Router

The router chooses an endpoint for a job using a cost model.

Generic cost form:

```text
cost(endpoint, job) = transfer_in
                    + queue_delay
                    + setup_or_launch
                    + compute
                    + transfer_out
```

The router selects:

```text
chosen_endpoint = argmin(cost(endpoint, job))
```

### SimPy Endpoint

The SimPy endpoint models a domain-specific accelerator. For this project, assume it represents a streaming vector/stencil accelerator with:

- moderate setup latency
- host-mediated transfer cost
- high throughput for supported operations
- limited concurrency
- queueing behavior
- operation-specific rates for SAXPY and Jacobi

It should be treated as a first-class endpoint in the cost model, even though actual numerical computation may be performed by CPU/GPU reference code.

---

## 5. Workloads

## 5.1 SAXPY MVP

SAXPY computes:

```text
y[i] = a * x[i] + y[i]
```

### Purpose

SAXPY is the simplest workload to validate:

- unified job API
- CPU/GPU/SIM endpoint support
- endpoint crossover behavior
- cost-model routing
- trace generation and SimPy replay

### Execution Modes

Run SAXPY under these policies:

1. Always CPU
2. Always GPU
3. Always SIM
4. Routed by cost model

### Expected Behavior

- Small `n`: CPU may win due to low overhead.
- Large `n`: GPU may win due to higher throughput.
- SIM may win in configured ranges depending on startup, bandwidth, queueing, and throughput parameters.
- Routed should track the best or near-best policy.

---

## 5.2 Jacobi Stencil POC

Jacobi stencil updates grid cells based on neighboring values.

For a 2D 5-point stencil:

```text
new[i][j] = c0 * old[i][j]
          + c1 * old[i-1][j]
          + c2 * old[i+1][j]
          + c3 * old[i][j-1]
          + c4 * old[i][j+1]
```

### Purpose

Jacobi is used to demonstrate the distributed/communication-aware extension of the routing layer.

### Two-Rank Setup

If two physical nodes are unavailable, run two MPI ranks on one physical node:

```bash
mpirun -np 2 ./jacobi_mpi
```

Treat:

```text
Rank 0 = logical node 0
Rank 1 = logical node 1
```

Each rank owns one subdomain and exchanges halo boundaries with the other rank.

### Per-Iteration Flow

```text
for each timestep:
    1. pack halo
    2. start halo exchange
    3. route INTERIOR_UPDATE among CPU/GPU/SIM
    4. wait for halo exchange
    5. unpack halo
    6. route BOUNDARY_UPDATE among CPU/GPU
    7. swap old/new grids
```

### Candidate Endpoints

```text
INTERIOR_UPDATE -> CPU, GPU, SIM
BOUNDARY_UPDATE -> CPU, GPU
HALO_EXCHANGE   -> MPI only
```

SIM should primarily accelerate `JACOBI_INTERIOR`, not the whole Jacobi iteration.

---

## 6. Trace-Driven Simulation Design

The project uses trace-driven simulation for the SIM endpoint.

### C++ Responsibility

The C++ runtime should:

1. Run CPU/GPU workloads and measure their costs.
2. Apply the router’s cost model.
3. Record the router’s endpoint choice.
4. Emit raw job traces.
5. Emit candidate cost estimates.

### Python/SimPy Responsibility

The Python SimPy model should:

1. Read raw traces.
2. Filter or identify jobs routed to SIM, or simulate SIM candidate costs for all jobs if desired.
3. Model queueing and service time.
4. Emit simulated timing/resource traces.
5. Optionally produce endpoint summary statistics.

### Evaluation Responsibility

The evaluation script should:

1. Merge raw C++ traces and SimPy results.
2. Compute oracle/best endpoint after measurements and simulation.
3. Compare router decision against oracle.
4. Compute prediction error.
5. Generate plots.

---

## 7. Raw Trace Schema

Use JSONL: one JSON object per line.

### SAXPY Example

```json
{
  "job_id": 12,
  "workload": "SAXPY",
  "op": "SAXPY",
  "arrival_us": 240.0,
  "n": 1048576,
  "input_bytes": 8388608,
  "output_bytes": 4194304,

  "cpu_measured_us": 180.5,
  "gpu_measured_us": 64.2,

  "cpu_estimated_us": 175.0,
  "gpu_estimated_us": 70.0,
  "sim_estimated_us": 85.0,

  "router_choice": "GPU",
  "endpoint": "SIM0",
  "metadata": {
    "dtype": "float32",
    "layout": "contiguous"
  }
}
```

### Jacobi Example

```json
{
  "job_id": 45,
  "workload": "JACOBI",
  "iteration": 3,
  "rank": 0,
  "node_id": 0,
  "endpoint": "SIM0",
  "op": "JACOBI_INTERIOR",
  "arrival_us": 3100.0,
  "nx": 2048,
  "ny": 1024,
  "halo_width": 1,
  "input_bytes": 8396800,
  "output_bytes": 8388608,

  "cpu_measured_us": 920.0,
  "gpu_measured_us": 260.0,
  "halo_measured_us": 80.0,

  "cpu_estimated_us": 900.0,
  "gpu_estimated_us": 280.0,
  "sim_estimated_us": 230.0,

  "router_choice": "SIM0",
  "metadata": {
    "stencil": "2d_5pt",
    "dtype": "float32",
    "layout": "row_major"
  }
}
```

---

## 8. SimPy Output Schema

The SimPy model should emit one record per simulated job.

```json
{
  "job_id": 45,
  "endpoint": "SIM0",
  "op": "JACOBI_INTERIOR",
  "arrival_us": 3100.0,
  "start_us": 3120.0,
  "finish_us": 3378.0,
  "queue_us": 20.0,

  "transfer_in_us": 40.0,
  "setup_us": 8.0,
  "compute_us": 190.0,
  "transfer_out_us": 40.0,

  "service_us": 278.0,
  "latency_us": 278.0,

  "capacity": 1,
  "endpoint_busy_us": 238.0,
  "endpoint_utilization": 0.73
}
```

Important distinction:

- `service_us` excludes queueing.
- `latency_us = finish_us - arrival_us` includes queueing.

---

## 9. Final Evaluation Trace Schema

After merging C++ and SimPy traces, emit a final record.

```json
{
  "job_id": 45,
  "workload": "JACOBI",
  "op": "JACOBI_INTERIOR",
  "rank": 0,

  "router_choice": "SIM0",

  "cpu_measured_us": 920.0,
  "gpu_measured_us": 260.0,
  "sim_estimated_us": 230.0,
  "sim_latency_us": 278.0,

  "best_endpoint": "GPU",
  "router_was_optimal": false,
  "prediction_error_us": 48.0
}
```

### Oracle Endpoint

The oracle endpoint is computed after all measurements/simulations are available:

```text
best_endpoint = argmin(cpu_measured_us, gpu_measured_us, sim_latency_us)
```

This allows evaluation of routing quality.

---

## 10. SimPy Model Parameters

The SimPy endpoint should model parameters that affect routing decisions.

Recommended configuration:

```python
@dataclass(frozen=True)
class SimEndpointConfig:
    setup_us: float
    bandwidth_bytes_per_us: float
    saxpy_elements_per_us: float
    jacobi_cells_per_us: float
    max_concurrency: int
```

Additional useful future parameters:

```text
jacobi_boundary_cells_per_us
tile_capacity_bytes
batching_enabled
batch_max_wait_us
input_latency_us
output_latency_us
```

### Sim Timing Formula

For SAXPY:

```text
T_sim = queue_delay
      + input_bytes / bandwidth
      + setup_us
      + n / saxpy_elements_per_us
      + output_bytes / bandwidth
```

For Jacobi interior:

```text
T_sim = queue_delay
      + input_bytes / bandwidth
      + setup_us
      + interior_cells / jacobi_cells_per_us
      + output_bytes / bandwidth
```

Interior cells:

```text
interior_cells = max(nx - 2 * halo_width, 0)
               * max(ny - 2 * halo_width, 0)
```

---

## 11. Performance Counters

### Per-Job Counters

Each SimPy job should track:

```text
job_id
op_type
arrival_us
start_us
finish_us
queue_us
transfer_in_us
setup_us
compute_us
transfer_out_us
service_us
latency_us
input_bytes
output_bytes
work_units
```

### Per-Endpoint Counters

Each endpoint should summarize:

```text
num_jobs
makespan_us
total_busy_us
utilization
avg_latency_us
p95_latency_us
avg_queue_us
max_queue_us
total_input_bytes
total_output_bytes
effective_throughput
```

### Router Validation Counters

Final evaluation should compute:

```text
predicted_cost_us
simulated_cost_us
prediction_error_us
prediction_error_pct
router_choice
best_endpoint
router_was_optimal
routing_accuracy
```

---

## 12. Cost Model

### Generic Form

```text
cost = transfer_in
     + queue_delay
     + setup_or_launch
     + compute
     + transfer_out
```

### CPU Cost

```text
cost_cpu = cpu_queue
         + cpu_fixed
         + cpu_per_work_unit * work_units
         + move_to_host_if_needed
```

### GPU Cost

```text
cost_gpu = gpu_queue
         + h2d_if_needed
         + cuda_launch
         + gpu_per_work_unit * work_units
         + d2h_if_needed
```

### SIM Cost

```text
cost_sim = sim_queue
         + modeled_transfer_in
         + sim_setup
         + sim_per_work_unit * work_units
         + modeled_transfer_out
```

### Jacobi Iteration Model

If halo exchange overlaps with interior update:

```text
T_iter = max(T_halo_exchange, T_interior_update)
       + T_boundary_update
       + T_sync
```

For two ranks:

```text
T_iter_global = max(T_iter_rank0, T_iter_rank1)
```

---

## 13. Implementation Plan

### Phase 1: Core C++ Types

Implement:

```text
Job
Endpoint
CostModel
Router
TraceWriter
```

Job kinds:

```text
SAXPY
JACOBI_INTERIOR
JACOBI_BOUNDARY
JACOBI_HALO_BOUNDARY
```

Endpoint kinds:

```text
CPU
GPU
SIM
```

### Phase 2: SAXPY Executors

Implement:

1. CPU SAXPY
   - Plain loop or OpenMP
   - Measure with `std::chrono`

2. GPU SAXPY
   - CUDA kernel wrapper
   - Measure with CUDA events

3. SIM SAXPY
   - Emit trace
   - Optional CPU reference computation for correctness

### Phase 3: SimPy Replay

Implement or extend:

```text
sim_endpoint.py
```

Functions:

```text
read_jsonl(path)
write_jsonl(path)
SimEndpointModel.simulate(jobs)
SimEndpointModel.endpoint_summaries(results)
```

### Phase 4: SAXPY Evaluation

Run:

```text
always CPU
always GPU
always SIM
routed
```

Generate:

```text
saxpy_raw.jsonl
saxpy_sim.jsonl
saxpy_final.jsonl
```

Plots:

```text
latency vs n
endpoint choice vs n
prediction error
SIM utilization
```

### Phase 5: Jacobi POC

Implement:

1. One-process or two-MPI-rank Jacobi baseline.
2. Two logical subdomains if physical two-node MPI is unavailable.
3. Halo exchange with MPI or modeled halo cost.
4. Interior/boundary job decomposition.
5. Trace emission per rank/iteration.

Generate:

```text
jacobi_raw.jsonl
jacobi_sim.jsonl
jacobi_final.jsonl
```

Plots:

```text
iteration time vs grid size
breakdown: halo/interior/boundary
endpoint utilization
routing accuracy
```

---

## 14. Repository Structure

Suggested layout:

```text
project/
  cpp/
    include/
      job.hpp
      endpoint.hpp
      cost_model.hpp
      router.hpp
      trace_writer.hpp
    src/
      saxpy_cpu.cpp
      saxpy_cuda.cu
      jacobi_cpu.cpp
      jacobi_cuda.cu
      cost_model.cpp
      router.cpp
      trace_writer.cpp
    bench/
      run_saxpy.cpp
      run_jacobi.cpp

  sim/
    sim_endpoint.py
    replay.py
    summarize.py

  eval/
    merge_traces.py
    plot_saxpy.py
    plot_jacobi.py

  traces/
    saxpy_raw.jsonl
    saxpy_sim.jsonl
    saxpy_final.jsonl
    jacobi_raw.jsonl
    jacobi_sim.jsonl
    jacobi_final.jsonl

  results/
    saxpy_latency_vs_n.png
    saxpy_endpoint_choice.png
    jacobi_iteration_time.png
    sim_utilization.png
    routing_accuracy.png
```

---

## 15. Correctness Strategy

### CPU/GPU

For SAXPY and Jacobi, compare CPU and GPU numerical outputs against a CPU reference.

### SIM

SimPy does **not** perform actual numerical computation. It models timing and resource behavior.

For jobs routed to SIM, use one of these strategies:

1. **Timing-only mode**
   - SimPy models timing.
   - Numerical output is not validated for SIM jobs.

2. **CPU-reference mode**
   - SimPy models timing.
   - CPU reference computes actual output for correctness.

For the report, clearly state:

> The simulated endpoint models performance behavior. Functional correctness is supplied by CPU/GPU reference implementations.

---

## 16. Evaluation Metrics

Use these primary metrics:

1. Endpoint latency
2. Estimated vs measured/simulated cost
3. Router choice
4. Oracle/best endpoint
5. Router accuracy
6. SIM queue delay
7. SIM utilization
8. Jacobi iteration time
9. Halo/interior/boundary breakdown
10. Prediction error

### Important Rule

Do **not** use SimPy wall-clock runtime as a performance metric.

Use:

```text
SimPy env.now / simulated time
modeled service time
modeled queue delay
modeled utilization
```

---

## 17. Expected Results and Interpretation

### SAXPY

Expected crossover behavior:

```text
small n  -> CPU likely wins
medium n -> SIM may win depending parameters
large n  -> GPU or SIM may win depending parameters
```

A good result is not necessarily that routed always wins. A good result is that routed follows the best endpoint across regimes and avoids clearly bad choices.

### Jacobi

Expected behavior:

- Small grids may not benefit from GPU/SIM due to setup/transfer overhead.
- Large interior regions may benefit from GPU/SIM.
- Halo exchange can dominate iteration time.
- Faster interior update may not help when halo exchange is the critical path.
- Communication-aware routing should explain when distribution helps or hurts.

---

## 18. How to Present the Novelty

Do not claim to invent heterogeneous scheduling. Existing systems such as StarPU, Legion, and PaRSEC already explore heterogeneous scheduling.

Claim a narrower contribution:

> This project prototypes a lightweight endpoint-routing layer that treats real CPU/GPU executors and trace-driven simulated domain-specific hardware as comparable physical endpoints under a shared cost model.

Suggested contribution list:

1. A unified job/endpoint API for heterogeneous computation.
2. A lightweight cost-model router for CPU/GPU/SIM endpoint selection.
3. A trace-driven SimPy endpoint model that reports latency, queueing, transfer, compute, and utilization.
4. An evaluation from SAXPY endpoint crossover to Jacobi stencil/halo communication-aware routing.

---

## 19. Known Limitations

1. SimPy endpoint is high-level, not cycle-accurate.
2. SIM does not perform actual computation unless paired with CPU reference execution.
3. Two-node behavior may be modeled using two MPI ranks or two logical subdomains if physical nodes are unavailable.
4. Only one real GPU may be available, so GPU should be modeled as a shared capacity-limited endpoint in distributed experiments.
5. Cost model is heuristic and should be presented as lightweight/prototype-level.
6. Results depend on configured SIM parameters.

These limitations are acceptable if clearly stated.

---

## 20. Immediate Next Tasks

1. Finalize raw trace schema.
2. Update `sim_endpoint.py` to distinguish:
   - `service_us`
   - `latency_us`
   - queue time
3. Add endpoint summary statistics:
   - average latency
   - p95 latency
   - average queue delay
   - utilization
4. Implement C++ trace writer.
5. Implement CPU SAXPY and GPU SAXPY timing.
6. Emit SAXPY raw traces.
7. Run SimPy replay and generate final SAXPY evaluation.
8. Implement Jacobi baseline and trace emission.
9. Add final evaluation scripts and plots.

---

## 21. One-Sentence Project Description

This project builds a cost-based endpoint-routing runtime that maps logical jobs to CPU, GPU, or SimPy-modeled domain-specific accelerator endpoints, using trace-driven simulation to evaluate specialized hardware behavior and routing quality on SAXPY and Jacobi stencil workloads.

