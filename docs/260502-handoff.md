# 2026-05-02 Routing Runtime Handoff

This note records the implementation progress for running Jacobi over two
logical nodes on one physical machine. The target experiment is constrained by
available hardware: one physical CPU and, when CUDA is visible, one physical
GPU. The runtime therefore models two logical nodes:

```text
Logical node 0: CPU0, optional GPU0, SIM0
Logical node 1: CPU1, SIM1
```

The important interpretation is that CPU0 and CPU1 are logical CPU endpoints
sharing the same host CPU. SIM0 and SIM1 are modeled endpoints in the offline
SimPy replay. GPU0 is exposed only when the process is explicitly allowed to
use the single physical GPU.

## Summary of Code Changes

### Endpoint Availability

Added `EndpointAvailability` to `include/router.h`:

```cpp
struct EndpointAvailability {
    bool cpu = true;
    bool gpu = true;
    bool sim = true;
};
```

`RouterConfig` now has:

```cpp
std::unordered_map<int, EndpointAvailability> endpoint_availability;
```

This lets each logical node declare which endpoints exist. The default remains
backward compatible: if a node is not listed, CPU/GPU/SIM are treated as
available, matching the old behavior.

### Router Behavior

Updated `src/router.cpp` so Auto and forced policies respect endpoint
availability.

For a target node, the router now checks whether CPU, GPU, and SIM are
available before estimating or choosing them. Unavailable endpoints receive an
internal large sentinel cost and are logged as `null` by the runtime.

Jacobi routing is now operator-based:

```text
JACOBI_INTERIOR      -> CPU / GPU / SIM, depending on availability
JACOBI_BOUNDARY      -> CPU / SIM, though the two-node example forces CPU
JACOBI_HALO_BOUNDARY -> CPU / SIM
```

GPU Jacobi remains supported only for `JACOBI_INTERIOR`, because the CUDA path
only implements the interior kernel.

`ForceSplit` for Jacobi now throws a clear error:

```text
Jacobi split dispatch is not implemented; route decomposed operators instead
```

This is intentional. We did not add `CPU+SIM` or `GPU+SIM` dispatch kinds
because Jacobi should first be decomposed into routable operators. Intra-operator
splitting would require new range semantics, concurrent CPU/SIM execution, and
write-safety rules.

### Cost Model Presets

Cost estimation and heuristics now live in the router layer, not in examples.
The public router API has:

```cpp
enum class CostModelPreset {
    Default,
    SimPyAlignedStencil
};

RouterConfig make_router_config(
    int local_node_id = 0,
    CostModelPreset preset = CostModelPreset::Default);

void apply_cost_model_preset(
    RouterConfig& config,
    CostModelPreset preset);
```

The example only selects a preset and declares endpoint availability. It does
not define per-byte or per-work-unit cost constants.

### Runtime Submit Overload

Added a runtime API overload in `include/runtime.h` and `src/runtime.cpp`:

```cpp
Request submit(Job job, RoutingPolicy policy, RouterConfig config);
```

The original call still works:

```cpp
Request submit(Job job, RoutingPolicy policy = RoutingPolicy::Auto);
```

The new overload is used by the two-logical-node Jacobi example so each logical
node can pass its own `local_node_id` and endpoint availability.

### Distributed Runtime Wrapper

Added `Routing::DistributedRuntime` in `include/runtime.h`. It hides MPI from
examples:

```cpp
Routing::DistributedRuntime runtime(&argc, &argv);
runtime.exchange_jacobi_halos(...);
runtime.gather_jacobi_global(...);
```

The example code does not include `mpi.h` and does not call `MPI_Send`,
`MPI_Recv`, `MPI_Sendrecv`, `MPI_Gather`, or `MPI_Gatherv`. Those details are
inside the runtime and CPU kernel layer.

`DistributedRuntime` initializes/finalizes MPI when needed, exposes logical
rank/node identity, exchanges halo rows, and gathers final owned rows back to
rank 0 for verification.

### CPU MPI Halo Kernel

Added `jacobi_exchange_halos_cpu(...)` in `src/kernels_cpu.cpp`. It performs the
two-rank halo exchange with `MPI_Sendrecv`.

For the current vertical two-node partition:

```text
node 0 sends its bottom owned row to node 1's top ghost row
node 1 sends its top owned row to node 0's bottom ghost row
```

This keeps communication in the kernel/runtime layer rather than in examples.

### Data Residency Tracking

Added a lightweight runtime data-residency table keyed by host buffer pointer:

```cpp
void mark_data_location(const void* buffer, DataLocation location);
DataLocation lookup_data_location(const void* buffer);
```

Before planning, `submit(...)` consults the table and fills job metadata
locations from tracked state when known. After execution, current endpoint
wrappers mark output buffers as `Host`, because today's CPU/GPU/SIM
implementations all leave user-visible vectors on host memory. This is
intentionally conservative: GPU currently copies back to host, and SIM runs CPU
fallback after emitting a trace.

The table gives the runtime a real place to evolve data placement later without
putting residency logic in examples.

### Per-Logical-Node Queue Pressure

Replaced the global runtime counters:

```cpp
g_cpu_jobs
g_gpu_jobs
g_sim_jobs
```

with a pressure table keyed by `target_node_id`:

```cpp
struct EndpointPressure {
    std::size_t cpu_jobs = 0;
    std::size_t gpu_jobs = 0;
    std::size_t sim_jobs = 0;
};

std::unordered_map<int, EndpointPressure> g_endpoint_pressure;
```

This means node 0 and node 1 can have separate logical CPU/SIM pressure even
inside one physical process. This is still a lightweight approximation, not a
full scheduler.

### JSON Logging for Unavailable Costs

The runtime now logs unavailable endpoint costs as JSON `null` instead of
`inf` or a huge numeric sentinel. Example:

```json
"cost_model":{"cpu":19.84,"gpu":null,"split":null,"sim":9.016,"remote":0}
```

This keeps JSONL output parseable and makes missing endpoints explicit.

## New Example: Two Logical Nodes

Added:

```text
examples/jacobi_two_logical_nodes.cpp
```

The example:

1. Builds a global Jacobi grid.
2. Splits interior rows across two logical nodes.
3. Gives each logical node a local tile with one ghost row on each side.
4. Exchanges halos by copying rows between the two local tiles.
5. Routes interior jobs with `RoutingPolicy::Auto`.
6. Forces boundary jobs to CPU.
7. Swaps buffers each iteration.
8. Gathers the final global grid.
9. Verifies against a sequential reference Jacobi implementation.

The logical per-iteration flow is:

```text
exchange_halos(node0, node1)
route node0 JACOBI_INTERIOR with Auto
route node1 JACOBI_INTERIOR with Auto
route node0 JACOBI_BOUNDARY with ForceCpu
route node1 JACOBI_BOUNDARY with ForceCpu
swap local buffers
```

The current example uses:

```text
nx = 64
ny = 34
iterations = 4
```

Each logical node owns 16 interior rows. With one ghost row above and below,
each local tile has:

```text
local nx = 64
local ny = 18
interior work units = (64 - 2) * (18 - 2) = 992
boundary work units = 160
```

## New MPI Example

Added:

```text
examples/jacobi_two_logical_nodes_mpi.cpp
```

This is the distributed version of the two-logical-node Jacobi test:

```bash
mpirun -np 2 ./build/bin/jacobi_two_logical_nodes_mpi
```

Each rank owns exactly one logical node:

```text
rank 0 -> logical node 0 -> CPU0, optional GPU0, SIM0
rank 1 -> logical node 1 -> CPU1, SIM1
```

The example still only expresses high-level runtime operations:

```text
construct distributed runtime
build local tile
exchange Jacobi halos through runtime
submit interior and boundary jobs
gather final grid through runtime
verify on rank 0
```

MPI implementation details are hidden behind `DistributedRuntime` and
`jacobi_exchange_halos_cpu`.

The latest validated MPI run produced:

```text
MPI two-logical-node Jacobi verification passed
Logical node 0 endpoints: CPU0, SIM0
Logical node 1 endpoints: CPU1, SIM1
```

Representative routing behavior:

```text
node 0 JACOBI_INTERIOR:
  node_kind: Local
  decision: SimOnly
  cpu: 19.84
  gpu: null
  sim: 9.016

node 1 JACOBI_INTERIOR:
  node_kind: Local
  decision: SimOnly
  cpu: 19.84
  gpu: null
  sim: 9.016
```

The important `node_kind` behavior is that each rank configures
`RouterConfig::local_node_id` to its own logical node ID. Therefore rank-local
jobs are planned as `Local`. If a rank submits a job for the other node, the
router will label it `Remote` and apply remote cost.

## GPU Exposure

The example now exposes GPU0 only when:

```bash
ROUTING_ENABLE_NODE0_GPU=1
```

Reason: the build machine can compile CUDA, but runtime execution reported:

```text
cudaMalloc(d_input): no CUDA-capable device is detected
```

So the safe default is:

```text
node 0: CPU0, SIM0
node 1: CPU1, SIM1
```

When running on a host where the physical GPU is visible, set
`ROUTING_ENABLE_NODE0_GPU=1` to model:

```text
node 0: CPU0, GPU0, SIM0
node 1: CPU1, SIM1
```

## Tuned Cost Model

The original two-node demo constants made SIM unrealistically cheap, especially
because SIM transfer cost was 1000x lower than GPU transfer cost. The tuned
router preset now uses microsecond-ish constants aligned with the offline SimPy
defaults. These constants are global endpoint heuristics. Workload-specific code
only supplies normalized work units and byte counts; the router decides which
endpoint terms apply.

In `src/router.cpp`, `CostModelPreset::SimPyAlignedStencil` applies:

```cpp
constexpr double pcie_bytes_per_us = 12000.0;

config.cpu_per_work_unit = 0.02;

config.cuda_launch = 6.0;
config.gpu_per_work_unit = 0.00008;
config.copy_to_gpu_per_byte = 1.0 / pcie_bytes_per_us;
config.copy_back_per_byte = 1.0 / pcie_bytes_per_us;

config.sim_setup = 8.0;
config.sim_per_work_unit = 1.0 / 4000.0;
config.sim_transfer_in_per_byte = 1.0 / pcie_bytes_per_us;
config.sim_transfer_out_per_byte = 1.0 / pcie_bytes_per_us;
```

The two-node example calls:

```cpp
Routing::RouterConfig config = Routing::make_router_config(
    local_node_id,
    Routing::CostModelPreset::SimPyAlignedStencil);
```

Interpretation:

```text
CPU compute: 0.02 us per cell
GPU launch: 6 us
GPU compute: 0.00008 us per cell
GPU transfer bandwidth: 12000 bytes/us
SIM setup: 8 us
SIM compute: 4000 cells/us
SIM transfer bandwidth: 12000 bytes/us
```

The important tuning change is that GPU and SIM now use the same modeled
transfer bandwidth. SIM is no longer winning because its transfer is
unreasonably cheap. It wins or loses based on setup and per-cell throughput.

## Dispatch Results After Tuning

### Safe Default, No GPU Exposed

Command:

```bash
./build/bin/jacobi_two_logical_nodes
```

Observed output:

```text
Two-logical-node Jacobi verification passed
Logical node 0 endpoints: CPU0, SIM0
Logical node 1 endpoints: CPU1, SIM1
```

Representative routing log entries:

```text
node 0 JACOBI_INTERIOR:
  decision: SimOnly
  cpu: 19.84
  gpu: null
  sim: 9.016

node 1 JACOBI_INTERIOR:
  decision: SimOnly
  cpu: 19.84
  gpu: null
  sim: 9.016

node 0 JACOBI_BOUNDARY:
  decision: CpuOnly
  cpu: 3.2
  gpu: null
  sim: 8.2

node 1 JACOBI_BOUNDARY:
  decision: CpuOnly
  cpu: 3.2
  gpu: null
  sim: 8.2
```

This is the correct behavior for the safe default. SIM beats CPU for interior,
and CPU beats SIM for boundary.

### GPU-Exposed Model

Command attempted:

```bash
ROUTING_ENABLE_NODE0_GPU=1 ./build/bin/jacobi_two_logical_nodes
```

The run failed on this machine because CUDA could not find a device, but the
partial routing log showed the intended cost model behavior:

```text
node 0 JACOBI_INTERIOR:
  decision: GpuOnly
  cpu: 19.84
  gpu: 6.84736
  sim: 9.016

node 1 JACOBI_INTERIOR:
  decision: SimOnly
  cpu: 19.84
  gpu: null
  sim: 9.016
```

This is the desired conceptual result:

```text
node 0 has GPU, so GPU wins local interior work
node 1 has no GPU, so SIM wins interior work
```

## SimPy Replay Results

After running:

```bash
eval "$(conda shell.zsh hook)"
conda activate drl
python -m prototype.run_simpy \
  --input outputs/sim_traces/routing_jacobi_two_node_sim_jobs.jsonl \
  --output outputs/sim_traces/routing_jacobi_two_node_sim_results.jsonl \
  --summary-output outputs/routing_jacobi_two_node_sim_summary.jsonl
```

The safe-default two-node run emitted 8 SIM jobs:

```text
4 jobs on SIM0
4 jobs on SIM1
```

Representative summary:

```json
{"busy_us":36.064,"capacity":1,"endpoint":"SIM0","makespan_us":11211.591,"utilization":0.0032166710326839427}
{"busy_us":36.064,"capacity":1,"endpoint":"SIM1","makespan_us":11222.327,"utilization":0.0032135937582285744}
```

Each SIM interior job has modeled service time:

```text
transfer_in_us  = 4608 / 12000 = 0.384
setup_us        = 8
compute_us      = 992 / 4000 = 0.248
transfer_out_us = 4608 / 12000 = 0.384
total_us        = 9.016
```

This matches the tuned router SIM estimate.

## Comparison With One-Node Jacobi Example

The existing one-node `examples/jacobi.cpp` is a small validation driver. It
does not benchmark Auto routing. It forces CPU first, then forces SIM to emit
trace records.

Latest representative one-node log:

```text
JACOBI_INTERIOR ForceCpu:
  decision: CpuOnly
  cpu: 140
  gpu: 1796
  sim: 6048

JACOBI_BOUNDARY ForceCpu:
  decision: CpuOnly
  cpu: 52
  gpu: null
  sim: 5416

JACOBI_INTERIOR ForceSim:
  decision: SimOnly
  cpu: 140
  gpu: 1796
  sim: 6048

JACOBI_BOUNDARY ForceSim:
  decision: SimOnly
  cpu: 52
  gpu: null
  sim: 5416

JACOBI_HALO_BOUNDARY ForceSim:
  decision: SimOnly
  cpu: 52
  gpu: null
  sim: 5416
```

Those costs use the default `RouterConfig`, not the tuned two-node demo config.
So do not compare the one-node costs directly against the tuned two-node costs
as if they were the same experiment.

## Conceptual Hardware Model for SIM

The SIM endpoint should be described as:

> A stencil-specialized streaming accelerator, similar to an FPGA or
> near-memory accelerator. It has setup/configuration latency, DMA-style
> transfers, one modeled pipeline/resource, and fixed throughput in cells per
> microsecond.

This is not a generic faster CPU. It represents hardware that is good at:

```text
regular stencil updates
contiguous tiles
streaming rows through line buffers
fixed coefficients
predictable throughput after startup
```

It is bad at:

```text
irregular control flow
tiny tiles where setup dominates
dynamic synchronization
arbitrary kernels
complex boundary handling
```

That story matches why `JACOBI_INTERIOR` is the best target for SIM while
boundary work can reasonably stay on CPU.

## Real-World Examples and References

These are useful references for motivating the SIM endpoint.

1. Multi-FPGA stencil acceleration, Tohoku University / IEEE Access.

   The paper describes pipelined FPGA accelerators for stencil computation and
   scaling across multiple FPGAs using high-speed QSFP links. The public summary
   reports up to 950 GFLOP/s on one FPGA and nearly doubled performance on two
   FPGAs, with competitive power efficiency compared with high-end GPUs.

   URL: https://tohoku.elsevierpure.com/en/publications/multi-fpga-accelerator-architecture-for-stencil-computation-explo/

2. SASA: Scalable and Automatic Stencil Acceleration on HBM-based FPGAs.

   SASA uses an analytical model to choose spatial and temporal parallelism for
   iterative stencil kernels on HBM-based FPGAs. The public abstract reports an
   average 3.41x speedup and up to 15.73x speedup on a Xilinx Alveo U280 FPGA
   compared with a prior automatic stencil framework.

   URL: https://experts.umn.edu/en/publications/sasa-a-scalable-and-automatic-stencil-acceleration-framework-for-/

3. PIMS: Processing-in-memory accelerator for stencil computations.

   PIMS models an in-memory stencil accelerator implemented in the logic layer
   of 3D-stacked memory. The motivation is that large stencil workloads are
   memory-bound and generate excessive memory traffic. The public summary says
   PIMS reduces data movement by 48.25% on average and reduces bank conflicts by
   up to 65.55%.

   URL: https://www.pnnl.gov/publications/pims-lightweight-processing-memory-accelerator-stencil-computations

4. Multi-FPGA scalable stencil computation with constant memory bandwidth.

   This work describes a custom computing machine called a scalable streaming
   array for stencil computations using multiple FPGAs. It is relevant because
   it frames stencil acceleration as a streaming-array problem.

   URL: https://cir.nii.ac.jp/crid/1360285710396703616

## Current Limitations

1. The two-node example is single-process, not MPI.

   Halo exchange is modeled with host row copies between two local tile buffers.
   This is enough for logical-node correctness, but it is not a distributed MPI
   benchmark.

2. CPU0 and CPU1 share the same physical CPU.

   Queue pressure is tracked separately by logical node, but real CPU execution
   still consumes the same host CPU.

3. GPU execution requires a visible CUDA-capable device.

   The code can compile CUDA here, but this runtime environment did not expose a
   CUDA device. Use `ROUTING_ENABLE_NODE0_GPU=1` only on a machine where GPU0 is
   actually visible.

4. Data residency is still metadata-only.

   The cost model can skip movement costs if metadata says data is resident on
   GPU/SIM, but the runtime does not maintain a real data placement table.

5. Router and SimPy are aligned for the tuned two-node example only.

   The global default `RouterConfig` still uses older placeholder constants.
   The existing one-node example therefore shows different cost scales.

6. Boundary jobs are forced to CPU in the two-node example.

   The tuned model would sometimes let SIM compete, but the current operator
   story intentionally keeps boundary handling on CPU while evaluating SIM/GPU
   for interior stencil work.

## Commands Used for Validation

Build:

```bash
make all
```

Safe default two-node run:

```bash
./build/bin/jacobi_two_logical_nodes
```

MPI two-node run:

```bash
mpirun -np 2 ./build/bin/jacobi_two_logical_nodes_mpi
```

In the current sandbox, `mpirun` needed elevated local socket permission to
start the PMIx listener. The program itself passed once MPI could launch ranks.

SimPy replay:

```bash
eval "$(conda shell.zsh hook)"
conda activate drl
python -m prototype.run_simpy \
  --input outputs/sim_traces/routing_jacobi_two_node_sim_jobs.jsonl \
  --output outputs/sim_traces/routing_jacobi_two_node_sim_results.jsonl \
  --summary-output outputs/routing_jacobi_two_node_sim_summary.jsonl
```

Existing smoke checks:

```bash
./build/bin/saxpy_small
./build/bin/jacobi
python -m unittest prototype.test_sim_endpoint
```

## Next Suggested Step

The next clean implementation step is to add a true two-rank MPI driver:

```bash
mpirun -np 2 ./build/bin/jacobi_mpi
```

Each rank would own one logical node:

```text
rank 0 -> node 0 -> CPU0/GPU0/SIM0
rank 1 -> node 1 -> CPU1/SIM1
```

The current single-process example already has the logical partitioning,
endpoint availability, routing, trace emission, and verification structure.
An MPI version would replace the host row-copy halo exchange with real rank
communication and give a clearer distributed-runtime story.
