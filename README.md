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
