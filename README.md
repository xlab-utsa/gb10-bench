# gb10-bench

Benchmarks for the NVIDIA DGX Spark (GB10): CPU, GPU and both together.

| Program | What it is | Typical run |
| --- | --- | --- |
| `gb10_stress` | The main suite: nine workloads × four precisions × every device, sized up to 7B-model scale, each case averaged over 25 runs with load, compute and total time. | 2–3 days |
| `gb10_trace` | The same suite built with instrumentation. Runs selected cases once and writes a detailed event log (JSON Lines) for building animations. Run it by hand after the main run. | minutes |
| `gb10_sweep` | The original matmul / sum scaling sweep from 16 to 1 billion elements. | ~10 minutes |

All three draw their own plots (SVG); no Python is needed.

## Build

```
make            # gb10_sweep and gb10_stress
make trace      # gb10_trace
```

Needed: `g++`, `make`, and the CUDA toolkit (`nvcc` and cuBLAS; CUDA 13.0 on the GB10).
Recommended, so the CPU matmul uses a tuned library:

```
sudo apt install libopenblas-openmp-dev
make -B
```

Without `nvcc` on the PATH the build is CPU-only. Without OpenBLAS the CPU uses the
built-in kernels, which are correct but several times slower for fp32/fp16 matmul.
After changing any `make` variable, rebuild with `make -B`.

## gb10_stress

### Run it

```
./gb10_stress --quick                      # a few minutes: every workload, precision and device at small sizes
nohup ./gb10_stress > stress.log 2>&1 &    # the full run
tail -f stress.log
```

Run `--quick` first. It exercises every code path and prints a WARNING if two
devices disagree on a result. The full run writes `stress.csv` row by row; if it
is interrupted, start it again with the same command and it resumes where it
stopped (`--fresh` starts over). When it finishes it writes the plots.

### Workloads, in rising order of complexity

| Workload | Operation | Largest size |
| --- | --- | --- |
| `sum` | Sum of n values | n = 2³³ (8.6 billion) |
| `histogram` | Histogram of n values (256 bins; 16 for int4) | n = 2³³ |
| `stencil` | 5-point Jacobi on a batch of N×N grids, 8 or 16 steps per pass | 32 grids of 16384² |
| `softmax` | Row softmax over a vocabulary | 131072 rows × 32000 |
| `linear` | Tokens through a stack of weight matrices, `Y = Σ X·Wᵀ` | 2048 tokens, 112 matrices of 14336×4096 (6.6 billion weights) |
| `conv2d` | 3×3 convolution, same padding | 64 images, 512 channels, 256×256 |
| `attention` | `softmax(QKᵀ/√d)·V` with the L×L scores held in memory | 256 sequences, L = 16384, d = 128 |
| `flash_v1` | FlashAttention v1: same result, scores never stored; key/value tiles outside, state reloaded and renormalised at every step | 256 sequences, L = 32768, d = 128 |
| `flash_v2` | FlashAttention v2: query rows outside and independent, state kept local, one normalisation at the end | 256 sequences, L = 32768, d = 128 |

Each workload has five to seven sizes, smallest first (`task_levels()` in
`stress/common.hpp`). The three attention workloads use identical data, so their
results must agree with each other; the suite's checks rely on that.

### Precisions

| Name | Storage | Arithmetic |
| --- | --- | --- |
| `fp32` | float | float |
| `fp16` | half | Values are widened to float for the math. The GPU matmul runs natively in half (cuBLAS). |
| `int8` | 8-bit, real value = raw/127 | Integer multiply-accumulate into int32. Exponentials are computed in float on the scaled value. |
| `int4` | 4-bit, two per byte, real value = raw/8 | Unpacked to int8 on the fly, then as int8. For `linear` and `conv2d` the weights are 4-bit and the activations 8-bit, as in quantised models. |

### Devices and sweep order

The suite makes one clean pass over all workloads per device, single
architectures first and mixed ones after:

1. `cpu:x925` — threads confined to the Cortex-X925 cores
2. `cpu:a725` — threads confined to the Cortex-A725 cores
3. `gpu` — GPU alone
4. `cpu` — all CPU cores together
5. `cpu+gpu` — GPU and all CPU cores at once

Core types are detected at start-up and listed with their CPU numbers; a machine
with one core type gets `cpu`, `gpu`, `cpu+gpu`. `--devices` selects a subset
(`--devices cpu:x925,gpu`; `clusters` means all single-core-type devices), and
`--cpu-sets "big=0-4,10-14;little=5-9,15-19"` replaces the detected groups with
your own.

In `cpu+gpu` each case is split between the devices: the GPU takes the first
share of the units (tokens, rows, sequences, images, grids) and the CPU the rest,
concurrently. The share starts from what the `cpu` and `gpu` sweeps measured for
that case and is corrected by how fast each half really ran while sharing the
machine. If `cpu` and `gpu` results for a case are missing (because `--devices`
left them out), the split is even and the notes column says so.

### What one case is

A case is one workload, precision, device, size and pass count. It runs once
untimed (warm-up) and then `--runs` times (25 by default). Every run is:

| Phase | Contents |
| --- | --- |
| load | Allocate the device's working memory and copy the operands into it from a staging buffer (`cudaMalloc` + host→device copy on the GPU; allocation + parallel copy on the CPU). |
| compute | `iters` passes of the workload. |

The CSV has the mean, minimum, maximum and standard deviation of load and
compute, and the mean of their sum. The timing is two clock reads around each
phase. Nothing inside the kernels is instrumented in this build, and plotting,
checksums, temperature readings and staging-data generation all happen outside
the timed phases.

### How far it goes

For every device, workload and precision the suite climbs the sizes; at the
largest size that fits in memory it keeps going by multiplying the pass count by
four. Before each case it predicts the time for all runs from the previous case,
and ends the series when the prediction exceeds `--case-budget` (1800 s). A series
that reaches the pass-count stage therefore ends with a case of about 7 to 30
minutes.

With five devices, nine workloads and four precisions that is 180 series,
roughly 2 to 3 days in total. To shorten it, lower `--case-budget` (900 roughly
halves it) or select with `--devices`, `--tasks`, `--precisions`. A case that
overruns twice its budget is stopped and recorded with the runs it completed.

### Memory

Staging data plus one working copy per device must fit in `--mem-gb` (90 GiB)
and in 88% of the memory available at start-up; sizes that do not fit are left
out and reported. The largest `linear` case in fp32 needs about 74 GiB in
`cpu+gpu` (staging plus a copy of the weights for each device). `attention` at
L = 32768 does not fit on the CPU (4 GiB of scores per thread) and is left out,
which is the limit FlashAttention exists to remove.

If a CUDA allocation fails although memory looks free, the page cache is usually
holding it: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.

### Output

`stress.csv` columns: `task, precision, device, level, shape, iters, runs, units,
elements, bytes, work, load_s, compute_s, total_s`, the min/max/sd columns,
`cpu_load_s, gpu_load_s, cpu_compute_s, gpu_compute_s, gpu_share` (the per-device
times and split of a `cpu+gpu` case), `gops_per_s`, `cpu_threads`, `impl` (which
implementation ran: `blas` or `openmp` on the CPU, `cublas` or `cuda` on the GPU,
both joined by `+` for `cpu+gpu`), `checksum`, `temp_c`
(hottest thermal zone after the case), `unix_time`.

`work` counts operations: 2·m·n·k for a matrix product, one per element for a
sum, and so on; `gops_per_s` is `work / compute_s / 1e9`.

Plots: `stress_total.svg`, `stress_compute.svg`, `stress_load.svg`,
`stress_rate.svg`. One row per workload, one column per precision, one line per
device. Open them in a browser and hover a point for its numbers.
`./gb10_stress plot stress.csv` redraws them.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--tasks`, `--precisions`, `--devices` | all | What to run |
| `--cpu-sets SPEC` | detected | Define the per-core-type CPU devices yourself |
| `--runs N` | 25 | Measured runs per case |
| `--no-warmup` | | Do not discard a first untimed run |
| `--case-budget S` | 1800 | Seconds allowed for all runs of one case |
| `--mem-gb G` | 90 | Memory budget per case (GiB) |
| `--max-level N` | | Stop the size ladder at level N |
| `--no-repeat` | | Do not continue with more passes at the largest size |
| `--out FILE`, `--fresh`, `--no-plot` | `stress.csv` | Output |
| `--quick` | | Two smallest sizes, one pass each, 3 runs |

CPU threads follow `OMP_NUM_THREADS`.

### Reading the results

- The digest of every case is compared across devices. Integer pipelines must
  match exactly; float ones within 0.2% (1% for fp16). A mismatch prints a
  WARNING and is counted at the end.
- At start-up the GPU backend runs each cuBLAS routine it wants to use (half
  matmul, int8 matmul, batched float matmul) on a tiny problem and compares with
  a reference. A routine that is missing or disagrees is replaced by a plain CUDA
  kernel, and the `impl` column says `cuda` instead of `cublas`.
- `linear` on the GPU and the fp32/fp16 `linear` and attention products on the
  CPU are library code (cuBLAS, OpenBLAS). Everything else is the hand-written
  kernels in `stress/cpu.cpp` and `stress/gpu.cu`: straightforward, not tuned to
  the last percent. Compare precisions and devices within a workload rather
  than reading the absolute numbers as the hardware's ceiling.
- `attention` computes in float on both devices for every precision (the
  operands are widened first), so its precisions differ mainly in load time.

## gb10_trace

`gb10_trace` is the stress suite compiled with every tracing hook switched on. It
is a separate binary so that the benchmark build carries no instrumentation at
all. Its timings are not benchmark numbers: it waits for every kernel to finish
before launching the next, and every kernel thread reports where it ran.

### Run it

After the main run:

```
make trace
./gb10_trace                                   # every workload, precision and device once, at size level 1
./gb10_trace --level 3 --tasks flash_v2 --precisions fp16 --devices gpu,cpu+gpu
```

It follows the same sweep order as the suite (single core types, GPU, all cores,
CPU+GPU) and accepts the same selection options. For `cpu+gpu` cases it reuses
the split the benchmark run recorded in `stress.csv` (`--shares FILE` to point
elsewhere), so the trace shows the division of work that was measured. Output:
`trace.jsonl` (the event log) and `trace_cases.csv` (the cases it ran).

| Option | Default | Meaning |
| --- | --- | --- |
| `--level N` | 1 | Which size to trace (0 is the smallest) |
| `--runs N` | 1 | Runs per case |
| `--trace-out FILE` | `trace.jsonl` | Event log |
| `--shares FILE` | `stress.csv` | Benchmark results to take the `cpu+gpu` splits from |
| `--sample-ms N` | 20 | System sampling period; 0 switches it off |
| `--slice-us N` | 1000 | Longest CPU work span before it is cut |
| `--block-detail N` | 512 | Kernels with up to N thread blocks also log every block |

Trace sizes grow quickly: expect tens to a few hundred megabytes for everything at
level 1, and gigabytes for a large level with `--block-detail` raised.

### The file

One JSON object per line, in time order within each case. Every event has

- `t` — nanoseconds since the trace started
- `dur` — duration in nanoseconds (0 for an instant)
- `ev` — the event type
- `case`, `run`, `phase` — which case it belongs to (see `case.begin`), which run
  (−1 is a warm-up), and `load`, `compute`, `checksum` or `unload`

| `ev` | Meaning | Main members |
| --- | --- | --- |
| `meta` | First line | `format`, `version`, `host`, `command`, sampling settings |
| `topology.cpu` | One per logical CPU | `cpu`, `core` (e.g. Cortex-X925), `midr`, `capacity`, `max_khz`, `cluster_cpus`, `caches` (level, type, size, sharing) |
| `topology.memory` | DRAM | `total_bytes`, `available_bytes`, `page_bytes`, huge-page setting |
| `topology.gpu` | GPU capacities | `name`, `sm_count`, `max_threads_per_sm`, `registers_per_sm`, `shared_memory_bytes_per_sm`, `warp_size`, `l2_cache_bytes`, `clock_khz`, `memory_bus_bits`, compute capability |
| `topology.link` | CPU–GPU relation | `shared_memory`, a note on what is and is not observable |
| `topology.sensors` | Names of the thermal zones sampled | `thermal_zones`, `nvml` |
| `device.begin` | A sweep starts | `device`, `kind`, `cpus` (the CPUs threads are confined to), `cpu_threads` |
| `case.begin` / `case.end` | A case | `task`, `precision`, `device`, `level`, `shape`, `iters`, `units`, `gpu_units`, `input_bytes`, `work` |
| `phase` | One phase on one device | `device` (`cpu` or `gpu`), `units`, `tid`, `cpu` |
| `host.alloc` / `host.free` | Host memory | `bytes`, `addr`, `tid`, `cpu` |
| `cpu.work` | A span of work by one thread on one core | `tid`, `cpu`, `kernel` (e.g. `stencil`, `flash_v2.qtile`, `host.copy`), `first`, `last`, `items`, `bytes_read`, `bytes_written` |
| `cpu.blas` | An OpenBLAS call | `routine`, `m`, `n`, `k` |
| `gpu.alloc` / `gpu.free` | Device memory | `bytes`, `addr`, `dur` |
| `gpu.xfer` | A host↔device copy | `dir`, `bytes`, `host`, `device`, `tid`, `cpu` |
| `gpu.kernel` | One kernel launch, start to finish | `kernel` (with the storage type it was compiled for, e.g. `k_fl2<fp16>`), `threads`, `blocks`, `block_size`, `regs_per_thread`, `shared_bytes`, `local_bytes_per_thread`, `sms_used`, `device_cycles`, `bytes_read_est`, `bytes_written_est`, `launch_tid`, `launch_cpu` |
| `gpu.sm` | What one SM did in that launch | `sm`, `blocks`, `threads`, `busy_ns`, `peak_blocks`, `peak_threads`, `peak_registers`, `avg_blocks_resident` (32 time slices) |
| `gpu.block` | One thread block (small launches only) | `block`, `sm`, `threads` |
| `gpu.blas` | A cuBLAS call | `routine`, `m`, `n`, `k`, `batch` |
| `sample` | Machine state, every `--sample-ms` | `cpu_util_pct[]`, `cpu_khz[]`, `temps_c[]`, `mem_available_bytes`, `process_rss_bytes`, and from NVML when present `gpu_util_pct`, `gpu_temp_c`, `gpu_power_w`, `gpu_clock_mhz` |

A `cpu.work` span merges consecutive loop iterations of one kernel on one thread
and core, up to `--slice-us`; a new span starts when the thread moves to another
core. Capacity for an animation comes from the `topology.*` lines: compare
`peak_threads` and `peak_registers` of a `gpu.sm` event with
`max_threads_per_sm` and `registers_per_sm`, and `sample.mem_available_bytes`
with `topology.memory.total_bytes`.

### How the GPU part is measured, and its limits

Each kernel in `stress/gpu.cu` starts by recording, per thread block, the SM it
is on and the device clock at its first start and last end. The SM id comes from
the PTX special register `%smid`, which is not a documented interface. If `nvcc`
or the driver rejects it, build with `make -B trace TRACE_SM=-DTRACE_NO_SMID`;
all blocks are then attributed to SM 0.

What the trace cannot contain, because software cannot observe it:

- **Register contents.** It has the number of registers per thread each kernel
  was compiled to use and how many threads were resident on each SM, which
  gives register-file occupancy, not values. Nothing comparable exists for the
  CPU cores.
- **Physical link traffic.** CPU and GPU share one DRAM pool. The trace has every
  host↔device copy and an estimate of the bytes each kernel reads and writes,
  not which link or lane carried them. Mapping that onto NVLink-C2C in an
  animation is your assumption.
- **Inside the libraries.** cuBLAS and OpenBLAS calls appear as one event each.
  Which SM ran a cuBLAS kernel is not recorded.
- **A common SM clock.** The device clock is a per-SM counter. Converting it to
  trace time assumes the counters started together and tick at `clock_khz`.
  Treat `gpu.sm` and `gpu.block` times as exact within one SM and approximate
  across SMs.

### CUPTI

`make -B trace CUPTI=1` also records what the CUDA driver reports for every
kernel (cuBLAS's internal ones included), copy and memset, as `cupti.kernel`,
`cupti.memcpy` and `cupti.memset` events: device-side start and end, grid and
block dimensions, registers per thread, queue and submit times. These carry
their own timestamps but no `case` tag; match them to a case by time. It needs the
CUPTI library from the CUDA toolkit (`extras/CUPTI`; set `CUPTI_DIR=` if it is
elsewhere).

## gb10_sweep

`./gb10_sweep` runs `C = A·B`, `Σx` and both together for N = 4 … 32768 on the CPU,
the GPU and a static CPU+GPU split, and writes `results.csv` and `results.svg`.
`./gb10_sweep --help` lists its options. It reports the best of several timings
rather than a mean, and is kept for its direct matmul scaling curve.

## Emulated GPU

`make emu` builds `gb10_stress_emu` and `gb10_trace_emu`, in which `stress/gpu.cu`
is compiled as plain C++ against `stress/cuda_emu.hpp`: every CUDA kernel runs
unchanged on CPU threads, and cuBLAS is replaced by reference loops. It is slow
and its timings mean nothing; it exists to check the GPU code path and the trace
format on a machine with no GPU.

## What has been tested

- All nine workloads × four precisions × CPU, emulated GPU and the CPU+GPU split
  agree on their results (integer pipelines exactly), and the three attention
  workloads agree with each other, on x86-64 and on AArch64 under QEMU.
- `stress/gpu.cu` compiles with `nvcc` 12.0 in both the benchmark and the trace
  build, and links against cuBLAS and CUPTI.
- It has **not** been run on a GPU or with CUDA 13 by its author. The first
  `./gb10_stress --quick` on the GB10 is the real test of the CUDA kernels, the
  cuBLAS calls and the core-type detection.
