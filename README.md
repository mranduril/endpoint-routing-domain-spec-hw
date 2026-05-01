## 05/01/26
In the 2-node Jacobi setup, I would make the **SimPy endpoint represent a stencil-specialized streaming accelerator**, not a generic “faster CPU.”

That fits your story best because Jacobi stencil is regular, memory-streaming, and local-neighborhood based — exactly the kind of workload that could plausibly be implemented as an FPGA pipeline or ASIC stencil engine.

## Recommended SimPy role

Each node has:

```text
Node 0: CPU0 + GPU0 + SIM0
Node 1: CPU1 + GPU1 + SIM1
```

The SimPy process models a local domain-specific accelerator:

```text
SIM endpoint = stencil streaming engine
```

It does **not** replace MPI. It does **not** own the whole distributed program. It is just another execution endpoint that can run certain stencil-related jobs.

---

# Best hardware assumption: streaming stencil engine

Assume the simulated hardware is a **fixed-function or semi-programmable stencil pipeline**.

It supports operations like:

```text
out[i][j] = c0 * in[i][j]
          + c1 * in[i-1][j]
          + c2 * in[i+1][j]
          + c3 * in[i][j-1]
          + c4 * in[i][j+1]
```

For Jacobi, this is perfect.

## What the hardware is good at

It is good at:

* regular 1D/2D stencil update
* contiguous tiles
* streaming rows through line buffers
* repeated iterations with same coefficients
* high throughput after startup
* deterministic execution time

## What it is bad at

It is bad at:

* irregular boundaries
* dynamic control flow
* arbitrary kernels
* tiny tiles
* data not laid out contiguously
* complicated synchronization

That gives the router real decisions to make.

---

# Where SimPy fits in the Jacobi pipeline

A Jacobi iteration can be decomposed into jobs:

```text
1. Halo pack
2. Halo exchange
3. Interior update
4. Border update
5. Halo unpack / boundary handling
```

Now assign possible endpoints:

| Job               | CPU |   GPU | SimPy stencil engine |
| ----------------- | --: | ----: | -------------------: |
| Interior update   | yes |   yes |      yes, strong fit |
| Border update     | yes |   yes |      maybe, weak fit |
| Halo pack/unpack  | yes | maybe |            no / weak |
| MPI halo exchange | yes |    no |                   no |
| Coefficient setup | yes |    no |                maybe |

The cleanest SimPy role is:

> **SimPy accelerates the interior stencil update on each node’s local subdomain.**

That is the most believable and easiest to model.

---

# Hardware aspects you can simulate

## 1. Startup / configuration latency

The accelerator must be configured with:

* grid dimensions
* stencil coefficients
* input/output buffer addresses
* tile shape

So model:

```text
config_latency_us
```

This makes CPU preferable for tiny tiles.

---

## 2. Streaming throughput

After startup, it processes cells at a fixed rate:

```text
cells_per_us
```

For a tile with (N) interior cells:

```text
exec_time = config_latency + N / cells_per_us
```

This is the core SimPy service model.

---

## 3. Line-buffer / tile constraints

A real stencil engine would use line buffers or scratchpad memory.

Model:

* maximum tile width
* maximum tile height
* maximum tile cells
* chunking overhead if tile is too large

Example:

```text
if tile_bytes > local_buffer_capacity:
    split into chunks
    add chunk_boundary_overhead
```

This makes the simulated hardware realistic.

---

## 4. Host-mediated data movement

Assume the accelerator is attached through PCIe or a host-visible DMA interface.

So it has transfer costs:

```text
input_transfer_time  = input_bytes / input_bandwidth
output_transfer_time = output_bytes / output_bandwidth
```

For a stencil tile, input bytes are slightly larger than output bytes because you need halo rows/columns:

```text
input tile = interior + ghost boundary
output tile = updated interior
```

This is important because a stencil accelerator may be compute-fast but data-movement-limited.

---

## 5. Limited concurrency

Assume each SimPy endpoint has:

* one stencil pipeline, or
* two pipelines

Model it with a SimPy `Resource(capacity=1)` or `capacity=2`.

This creates queueing delay:

```text
sim_cost = queue_delay + transfer_in + config + compute + transfer_out
```

---

## 6. Supported stencil radius

Make it support only:

* radius-1 5-point stencil, or
* maybe radius-1 7-point if 3D later

For MVP:

```text
supported_ops = {JACOBI_2D_5PT_INTERIOR}
```

Unsupported jobs fall back to CPU/GPU.

That makes it domain-specific.

---

# How the router uses SimPy

For each stencil sub-job, the router estimates:

```text
cost_cpu(tile)
cost_gpu(tile)
cost_sim(tile)
```

Then picks the cheapest endpoint.

For interior tile:

```text
cost_sim =
    sim_queue_delay
  + input_bytes / sim_input_bw
  + sim_config_latency
  + num_cells / sim_cells_per_us
  + output_bytes / sim_output_bw
```

For GPU:

```text
cost_gpu =
    h2d_if_needed
  + cuda_launch_latency
  + num_cells / gpu_cells_per_us
  + d2h_if_needed
```

For CPU:

```text
cost_cpu =
    num_cells / cpu_cells_per_us
```

For two nodes, add halo communication outside the local endpoint cost:

```text
iteration_cost =
    local_compute_cost
  + halo_exchange_cost
  + sync_cost
```

Or, if you overlap interior compute with halo exchange:

```text
iteration_cost =
    max(interior_compute_cost, halo_exchange_cost)
  + border_compute_cost
  + sync_cost
```

That overlap formula is very useful for your report.

---

# Best decomposition for your POC

I would implement the Jacobi iteration like this:

```text
for each timestep:
    1. start halo exchange
    2. route INTERIOR_UPDATE job
    3. wait for halo
    4. route BORDER_UPDATE job
    5. swap buffers
```

Now SimPy can be useful:

```text
INTERIOR_UPDATE -> candidate endpoints: CPU, GPU, SIM
BORDER_UPDATE   -> candidate endpoints: CPU, GPU
HALO_EXCHANGE   -> MPI only
```

This is very elegant.

The SimPy endpoint does not need to handle border cells. That avoids boundary-condition complexity.

---

# Why this is better than making SimPy a whole-node accelerator

Do **not** make SimPy “run the whole Jacobi iteration.”

That would hide the compiler/runtime decision.

Instead, make it an endpoint for a specific operator:

```text
JACOBI_INTERIOR_TILE
```

Then your router has meaningful decisions:

* CPU for small border jobs
* GPU for large general tiles
* SIM for supported regular interior tiles
* MPI for halo exchange

This is much more compiler-like.

---

# What assumptions are “big-brain but realistic”?

Here is a strong assumption set:

## SimPy endpoint = FPGA-like stencil streaming engine

It has:

```text
supported op:
  2D 5-point Jacobi interior update

data layout:
  contiguous row-major float32

config latency:
  medium

throughput:
  high cells/sec after startup

memory:
  limited line buffer / scratchpad

transfer:
  host-mediated DMA-like input/output movement

concurrency:
  1 pipeline per node

strength:
  predictable throughput and low energy

weakness:
  poor for tiny jobs and unsupported boundary logic
```

This is plausible on FPGA or ASIC.

---

# Example parameter table

You can use something like this in the paper:

| Parameter | Meaning                                 |
| --------- | --------------------------------------- |
| `L_cfg`   | accelerator configuration latency       |
| `B_in`    | input bandwidth into accelerator        |
| `B_out`   | output bandwidth from accelerator       |
| `R_cell`  | stencil cells processed per microsecond |
| `C`       | max concurrent pipelines                |
| `M_local` | local scratchpad / tile capacity        |
| `Q`       | current queue delay                     |

Then:

```text
T_sim(tile) =
    Q
  + L_cfg
  + input_bytes / B_in
  + output_bytes / B_out
  + cells / R_cell
  + chunk_overhead(tile, M_local)
```

That is a very clean model.

---

# How this helps your 2-node story

With SimPy included, you can compare:

## Baseline 1

CPU-only Jacobi with MPI halo.

## Baseline 2

GPU-only Jacobi with MPI halo.

## Baseline 3

Fixed SimPy interior update.

## Your routed version

Cost model chooses:

* CPU/GPU/SIM for interior
* CPU/GPU for border
* MPI for halo

This shows the value of the routing layer.

Even if SimPy is artificial, your contribution is still real:

> the framework can incorporate a domain-specific endpoint by giving it an executor and a cost model.

That is exactly the point.

---

# What would be a good result?

A very believable result is:

* small grids: CPU wins because overhead dominates
* medium grids: GPU or SIM wins depending on parameters
* large grids: GPU/SIM wins for interior
* high SimPy queue: router avoids SIM and uses GPU
* high halo cost: two-node execution stops scaling

You do **not** need SimPy to always beat GPU. In fact, it is more credible if it does not.

The best result is:

> routed execution tracks the best fixed endpoint across regimes and avoids bad endpoint choices.

That is very defensible.

---

# Final recommendation

In your Jacobi setup, make SimPy model a:

> **stencil-specific streaming accelerator for regular interior tile updates**, with configurable setup latency, DMA bandwidth, pipeline throughput, scratchpad capacity, and queueing.

Then your router treats it as a first-class endpoint alongside CPU and GPU.

Use it only for:

```text
INTERIOR_STENCIL_UPDATE
```

and let CPU/GPU handle:

```text
BORDER_UPDATE
HALO_PACK/UNPACK
MPI coordination
```

That gives you the cleanest and most realistic story.


## Cost model assumption
cost(endpoint, job) = transfer_cost + queue_delay + execution_cost

cost_cpu(n) = alpha_cpu * n

cost_gpu(n) = launch_gpu + alpha_gpu * n + h2d_cost(n) + d2h_cost(n)

cost_sim(n) = startup_sim + alpha_sim * n + queue_delay + h2d_cost(n) + d2h_cost(n)

n: Problem size; alpha_cpu

## Project direction

I shifted from the original **GPU + Vortex FPGA + BlueField** idea to a more feasible **endpoint-routing runtime** project.

Core idea:

* A **job** is a logical unit of work.
* An **endpoint** is where the job can run.
* A **router** chooses the endpoint using a **cost model / heuristics**.
* This is analogous to **query optimization**:

  * logical job = logical operator
  * CPU / GPU / SIM endpoint = physical plans
  * execution + transfer + queue delay = cost

Important layering:

* **Frontend** = my routing API / job abstraction
* **Execution backends** = CPU function, CUDA wrapper, SimPy service
* **Communication backends** = MPI/HPC-X later, maybe NCCL later
* **CUDA / MPI / NCCL are not endpoints**
* **CPU / GPU / SIM are endpoints**

---

## Current hardware scope

I am no longer assuming Vortex is available.

Current practical scope:

* Start with **1 node**
* Endpoints:

  * **CPU**
  * **GPU**
  * **SimPy process** as simulated domain-specific hardware

Later POC:

* **2 nodes**
* Each node has:

  * 1 CPU endpoint
  * 1 GPU endpoint
  * 1 SimPy endpoint

---

## Why SimPy

The professor suggested replacing unavailable hardware with a **high-level Python emulator**.
We agreed SimPy is a good fit because it can model:

* startup latency
* throughput
* queueing
* limited concurrency
* bandwidth-ish service delay

This simulated endpoint should behave like a **service endpoint**, not necessarily real hardware.

---

## Workload decisions

We discussed crypto workloads like **Ed25519 / RSA**, but concluded:

* **Ed25519** is easier than RSA if I ever need crypto.
* But **distributed Ed25519 over 2 nodes is not a great main story**, unless it is a high-throughput batched service.
* For the **first MVP**, the least-effort useful job is **SAXPY**.

So:

* **Single-node MVP job**: **SAXPY**
* **Two-node POC workload**: **stencil / halo exchange**

Reason:

* SAXPY is trivial on CPU, GPU, and easy to model in SimPy.
* Stencil/halo is a much better 2-node communication-aware workload than distributed Ed25519.

---

## Programming model we settled on

Do **not** expose OpenMP/NCCL/CUDA directly as the programming model.

Instead:

* Programmer writes against **my routing API**
* Router chooses endpoint
* Endpoint-specific executors run the job:

  * CPU endpoint → OpenMP/C++ function
  * GPU endpoint → CUDA wrapper function around kernels
  * SIM endpoint → SimPy request/service

For communication later:

* MPI/HPC-X for inter-node
* CUDA memcpys locally
* NCCL only if needed later

---

## Job abstraction

A job should be an abstract work item with metadata, not raw CUDA code.

Example shape:

```text
Job {
    id
    type
    input buffers
    output buffers
    size / n
    metadata
    preferred targets
}
```

For SAXPY:

```text
Job {
    type = SAXPY
    scalar = a
    input_1 = x
    input_2 = y
    output = y
    n = vector_length
}
```

Endpoint handlers:

* CPU: `run_saxpy_cpu(...)`
* GPU: `run_saxpy_gpu(...)`
* SIM: `simpy_submit(...)`

Yes, CUDA code should be prepared as **host-callable wrapper functions**, not exposed directly to the router.

---

## Cost model direction

We agreed it is a good idea to start with simple heuristics.

Base model:

```text
cost(endpoint, job) =
    transfer_cost +
    queue_delay +
    execution_cost
```

For single-node SAXPY, something like:

```text
cost_cpu(n) = alpha_cpu * n
cost_gpu(n) = launch_gpu + alpha_gpu * n
cost_sim(n) = startup_sim + alpha_sim * n + queue_delay
```

Router:

* enumerate feasible endpoints
* score each endpoint
* pick argmin cost

This is the main “query optimization” analogy.

---

## Single-node MVP plan

### Goal

Show one logical job can be routed across CPU, GPU, and SIM endpoints using a simple cost model.

### Configuration

1 node:

* CPU endpoint
* GPU endpoint
* SimPy endpoint

### Job

* SAXPY only

### Minimum API

Something like:

```text
Job j = make_saxpy(a, x, y, n);
Result r = submit(j, policy=AUTO);
wait(r);
```

### Implementations

* CPU SAXPY: plain C++ / OpenMP
* GPU SAXPY: CUDA kernel + wrapper
* SimPy SAXPY: emulated service time only

### Compare

* always CPU
* always GPU
* always SIM
* routed

### Metrics

* latency vs `n`
* throughput vs request rate
* endpoint decisions vs `n`

---

## 2-node POC plan

### Goal

Make communication part of the optimization problem.

### Workload

Use **1D or 2D stencil / halo exchange**

### Why

Better fit for 2 nodes because:

* communication matters
* overlap matters
* placement matters
* halo / border / interior can be routed differently

### Jobs could become

* `INTERIOR_STENCIL`
* `BORDER_STENCIL`
* `HALO_PACK`
* `HALO_UNPACK`

### Cost model extension

```text
cost(endpoint, job) =
    transfer_cost +
    queue_delay +
    execution_cost +
    synchronization_penalty
```

### Baselines

* CPU-only stencil
* GPU-only stencil
* fixed placement
* routed placement

### Metrics

* iteration time
* communication/computation breakdown
* endpoint utilization
* overlap

---

## Existing related systems we found

This is **not** the first CPU/GPU heterogeneous runtime idea in general.

Existing comparable systems:

* **StarPU**
* **Legion**
* **PaRSEC**
* **Charm++**
* **Kokkos** is related at abstraction level

So novelty should be framed as:

* lightweight endpoint-routing runtime
* over CPU + GPU + simulated domain-specific endpoint
* with simple cost-based placement
* prototype-friendly and swappable with real hardware later

---

## What not to do in MVP

Avoid for now:

* real FPGA integration
* BlueField / DOCA dataplane
* UCX plugin work
* complex compiler integration
* packet/network simulation
* multi-job workload too early
* RSA/Ed25519 as first MVP workload

---

## Recommended timeline

### Week 1

* define job + endpoint abstractions
* implement CPU SAXPY
* implement GPU SAXPY
* basic router with thresholds

### Week 2

* add SimPy endpoint
* add timing + measurements
* run single-node experiments

### Week 3

* fit simple cost model
* compare fixed vs routed
* generate plots
* package MVP

### After that

* move to 2 nodes
* implement stencil/halo baseline
* extend router with communication cost

---

## Clean one-paragraph project description

We prototype a **cost-based endpoint-routing runtime** for heterogeneous systems. A logical job is mapped to one of several physical execution endpoints—CPU, GPU, or a simulated domain-specific hardware endpoint modeled in SimPy—using lightweight estimates of execution, transfer, and queueing costs. We first validate the idea on a **single-node SAXPY MVP**, then extend it to a **two-node stencil/halo proof of concept** where communication becomes a first-class optimization concern.

---

## Immediate next step

The immediate implementation target should be:

**Single-node SAXPY MVP with CPU + CUDA + SimPy endpoint and a simple argmin cost router.**

If you want, I can also turn this into a repo skeleton with filenames and module boundaries.
