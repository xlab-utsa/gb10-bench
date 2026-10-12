# gb10-bench

A scaling sweep for the NVIDIA DGX Spark (GB10): simple linear algebra, from tens of
elements to about a billion, on the CPU, the GPU, and both together. One C++/CUDA
source file; it writes a CSV and draws its own plots (SVG), with no Python needed.

## What it measures

| Workload | Operation | Work counted |
| --- | --- | --- |
| `matmul` | `C = A·B`, N×N float32 | 2N³ flops |
| `sum` | `s = Σx` over N² float32 values | N² adds |
| `matmul+sum` | `C = A·B`, then `s = ΣC`, timed end to end | 2N³ + N² |

| Device | How |
| --- | --- |
| `cpu` | All cores. OpenBLAS `sgemm` if installed, otherwise a built-in blocked OpenMP kernel; OpenMP sum. |
| `gpu` | cuBLAS `sgemm` and a CUDA reduction kernel. |
| `cpu+gpu` | Both at once. The GPU takes the first share of the columns (matmul) or elements (sum) and the CPU takes the rest, concurrently. The share is proportional to the two devices' measured solo rates at that size. |

The x axis is "elements" = N²: the size of one matrix, or the vector length for the
sum. The default grid runs N = 4 … 32768, which is 16 … 1.07 billion elements, with
two points per doubling of N.

## Build

```
make
```

`make` uses `nvcc` when it is on the PATH (CUDA 13.0 on the GB10) and falls back to a
CPU-only build with `g++` otherwise. Needed: `g++`, `make`, and the CUDA toolkit
(`nvcc` and cuBLAS). Recommended, for a tuned CPU matmul:

```
sudo apt install libopenblas-openmp-dev
make -B
```

Without OpenBLAS the CPU matmul uses the built-in kernel, which is correct but
several times slower, so the CPU curve will understate the machine.

## Run

```
./gb10_sweep --quick      # a few seconds: checks that everything works
./gb10_sweep              # the full sweep -> results.csv and results.svg
./gb10_sweep plot results.csv [out.svg]    # redraw the plots from a CSV
```

Open `results.svg` in a browser. It has one row per workload and three columns: time
per call, throughput in GFLOP/s, and speedup over the CPU. Hover a point for its
exact values. Each CSV row carries the size, time, GFLOP/s, repetitions, the GPU's
share of the work, and a checksum that must agree across devices (the program warns
if it does not).

| Option | Default | Meaning |
| --- | --- | --- |
| `--min-n`, `--max-n` | 4, 32768 | Range of the matrix side N |
| `--ppo` | 2 | Grid points per doubling of N |
| `--mem-gb` | 90 | Memory budget for all buffers, in GiB |
| `--budget` | 600 | End a series when its next point is predicted to take longer (seconds) |
| `--devices` | `cpu,gpu,both` | Which device series to run |
| `--workloads` | `matmul,sum,pipe` | Which workloads to run (`pipe` = matmul+sum) |
| `--out` | `results.csv` | CSV path; the SVG takes the same stem |
| `--no-plot` | | Skip the SVG |

CPU threads follow `OMP_NUM_THREADS`.

## Memory

All buffers are allocated once, at the largest N: three matrices on the host and
three on the device. On the GB10 both come out of the same DRAM, so the total is
6 · N² · 4 bytes, or 24 GiB at N = 32768. The plan is checked against `--mem-gb`
and against 85% of `MemAvailable` before anything is allocated; sizes that do not
fit are dropped from the grid. `--max-n 46341` (2.1 billion elements) needs 48 GiB.

If a CUDA allocation fails although memory looks free, the page cache is usually
holding it: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.

## Reading the numbers

- GPU times are compute-only. The inputs are already on the device, and the
  `cpu+gpu` result is left split between host and device memory (no merge copy).
- Small sizes are batched and reported as the best of several samples. A call that
  takes a second or more is run once.
- `cpu+gpu` always gives each device at least one unit of work, so at small sizes it
  pays the GPU's launch latency and tracks the GPU curve. Its `gpu_share` column
  shows how the work was divided.
- The header prints which OpenBLAS kernels were selected. If it reports generic
  `ARMV8` kernels on the GB10, setting `OPENBLAS_CORETYPE` (for example
  `NEOVERSEN2`) before the run may give a faster CPU matmul.
