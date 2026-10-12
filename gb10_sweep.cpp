// gb10_sweep.cpp - CPU / GPU / CPU+GPU scaling sweep for the NVIDIA DGX Spark (GB10).
//
// Three workloads, all float32, swept over a geometric grid of matrix sides N:
//   matmul      C = A * B                  (N x N, 2*N^3 flops)
//   sum         s = sum(x)                 (N*N elements, one add each)
//   matmul+sum  C = A * B, then s = sum(C) (timed end to end)
// on three devices:
//   cpu      all cores  (BLAS sgemm if linked, else the built-in OpenMP kernel)
//   gpu      cuBLAS sgemm + a CUDA reduction kernel
//   cpu+gpu  both at once: the GPU takes the first share of the columns
//            (matmul) or elements (sum), the CPU takes the rest, concurrently.
//            The share is proportional to the two devices' measured solo rates.
//
// "elements" (the x axis) is N*N: the size of one matrix, or the vector length
// for the sum. The default grid runs N = 4 .. 32768, i.e. 16 .. 1.07e9 elements.
//
// Memory: buffers are allocated once at the largest N. That is 3 host matrices
// plus 3 device matrices (the GB10 shares one DRAM pool between CPU and GPU, so
// both count): 6 * N^2 * 4 bytes = 24 GiB at N = 32768. The plan is checked
// against --mem-gb (default 90) and against MemAvailable before anything runs.
//
// GPU timings are compute-only: inputs are already on the device, and the
// cpu+gpu result is left split across host and device memory (no merge copy).
//
// Build:   make            (uses nvcc when present, plain g++ otherwise)
// Run:     ./gb10_sweep                 -> results.csv + results.svg
// Replot:  ./gb10_sweep plot results.csv [out.svg]
//
// The plotting code (section 5) only reads the CSV; it never runs during timing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// ============================================================================
// 1. Small utilities
// ============================================================================

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "error: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

template <class F> static double time_calls(F& f, long calls) {
    const double t0 = now_s();
    for (long i = 0; i < calls; ++i) f();
    return now_s() - t0;
}

struct Timing { double sec; long reps; };

// Best-of-several timing. The first call is a warm-up; a call that takes a
// second or more is too expensive to repeat and is reported as measured.
// Fast calls are batched so that each sample lasts at least a millisecond.
template <class F> static Timing measure(F&& f) {
    double t = time_calls(f, 1);
    if (t >= 1.0) return {t, 1};
    long inner = 1;
    while ((t = time_calls(f, inner)) < 1e-3 && inner < (1L << 26)) inner *= (t < 1e-4 ? 8 : 2);
    double best = t / inner, spent = t;
    long reps = inner;
    for (int s = 1; s < 9 && (s < 3 || spent < 0.25); ++s) {
        t = time_calls(f, inner);
        best = std::min(best, t / inner);
        spent += t;
        reps += inner;
    }
    return {best, reps};
}

static std::string fmt_time(double s, bool unicode = false) {
    const char* unit = "s";
    double v = s;
    const double e = 1.000001;  // keeps exact decade ticks in the larger unit
    if (s * e < 1e-6) { v = s * 1e9; unit = "ns"; }
    else if (s * e < 1e-3) { v = s * 1e6; unit = unicode ? "\xC2\xB5s" : "us"; }
    else if (s * e < 1.0) { v = s * 1e3; unit = "ms"; }
    char b[48];
    std::snprintf(b, sizeof b, "%.3g %s", v, unit);
    return b;
}

static std::string fmt_count(double v) {
    static const char* suffix[] = {"", "k", "M", "B", "T"};
    int i = 0;
    while (v >= 999.5 && i < 4) { v /= 1000.0; ++i; }
    char b[48];
    std::snprintf(b, sizeof b, "%.3g%s", v, suffix[i]);
    return b;
}

// ============================================================================
// 2. CPU kernels
// ============================================================================

static int cpu_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#ifdef USE_BLAS
extern "C" void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                       const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
                       const float* beta, float* c, const int* ldc);
extern "C" char* openblas_get_config(void) __attribute__((weak));
extern "C" char* openblas_get_corename(void) __attribute__((weak));
extern "C" int openblas_get_parallel(void) __attribute__((weak));  // 1 = pthreads, 2 = OpenMP
#endif

static std::string cpu_gemm_backend() {
#ifdef USE_BLAS
    if (!openblas_get_config) return "BLAS sgemm";
    std::istringstream cfg(openblas_get_config());  // "OpenBLAS 0.3.26 NO_LAPACKE ..."
    std::string name, version;
    cfg >> name >> version;
    std::string s = name + " " + version + " sgemm";
    if (openblas_get_corename) s += std::string(" (") + openblas_get_corename() + " kernels)";
    return s;
#else
    return "built-in OpenMP kernel";
#endif
}

// True when the linked BLAS runs its own pthread pool next to our OpenMP pool.
static bool blas_has_own_threads() {
#ifdef USE_BLAS
    return openblas_get_parallel && openblas_get_parallel() == 1;
#else
    return false;
#endif
}

// Fallback matmul, column-major: C[:, 0:nc] = A * B[:, 0:nc] with A being N x N.
// Cache-blocked; OpenMP over row blocks, SIMD along the rows of each block.
// gemm_rows computes rows [i0, i0 + len) of the result.
[[maybe_unused]] static void gemm_rows(int N, int nc, const float* __restrict__ A, const float* __restrict__ B,
                                       float* __restrict__ C, int i0, int len) {
    const int KB = 128;  // a len x KB block of A (<= 256 KB) stays in L2 across all columns
    for (int j = 0; j < nc; ++j) std::memset(C + (size_t)j * N + i0, 0, (size_t)len * sizeof(float));
    for (int k0 = 0; k0 < N; k0 += KB) {
        const int k1 = std::min(N, k0 + KB);
        for (int j = 0; j < nc; ++j) {
            float* __restrict__ c = C + (size_t)j * N + i0;
            const float* b = B + (size_t)j * N;
            int k = k0;
            for (; k + 4 <= k1; k += 4) {
                const float b0 = b[k], b1 = b[k + 1], b2 = b[k + 2], b3 = b[k + 3];
                const float* a0 = A + (size_t)k * N + i0;
                const float *a1 = a0 + N, *a2 = a1 + N, *a3 = a2 + N;
                #pragma omp simd
                for (int i = 0; i < len; ++i) c[i] += a0[i] * b0 + a1[i] * b1 + a2[i] * b2 + a3[i] * b3;
            }
            for (; k < k1; ++k) {
                const float b0 = b[k];
                const float* a0 = A + (size_t)k * N + i0;
                #pragma omp simd
                for (int i = 0; i < len; ++i) c[i] += a0[i] * b0;
            }
        }
    }
}

[[maybe_unused]] static void gemm_builtin(int N, int nc, const float* A, const float* B, float* C) {
    if ((double)N * N * nc < 3e5) {  // too small to be worth waking the thread pool
        gemm_rows(N, nc, A, B, C, 0, N);
        return;
    }
    const int IB = std::min(512, std::max(16, (N / (4 * cpu_threads()) + 15) / 16 * 16));
    const long nblk = (N + IB - 1) / IB;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long blk = 0; blk < nblk; ++blk) {
        const int i0 = (int)blk * IB;
        gemm_rows(N, nc, A, B, C, i0, std::min(N, i0 + IB) - i0);
    }
}

static void cpu_gemm(int N, int nc, const float* A, const float* B, float* C) {
#ifdef USE_BLAS
    const float one = 1.0f, zero = 0.0f;
    sgemm_("N", "N", &N, &nc, &N, &one, A, &N, B, &N, &zero, C, &N);
#else
    gemm_builtin(N, nc, A, B, C);
#endif
}

// Sum of one chunk. 32 independent partial sums keep the adds from forming one
// dependency chain; the compiler turns them into vector registers.
static double chunk_sum(const float* x, size_t len) {
    enum { L = 32 };
    float acc[L] = {0.0f};
    size_t i = 0;
    for (; i + L <= len; i += L)
        for (int j = 0; j < L; ++j) acc[j] += x[i + j];
    for (int w = L / 2; w >= 1; w /= 2)
        for (int j = 0; j < w; ++j) acc[j] += acc[j + w];
    float s = acc[0];
    for (; i < len; ++i) s += x[i];
    return s;
}

// Sum of n floats: float accumulation inside 8192-element chunks, double across
// chunks, so a billion-element sum stays accurate. Small inputs stay on one
// thread, where waking the pool would cost more than the sum itself.
static double cpu_sum(const float* x, size_t n) {
    const size_t chunk = 8192;
    const long nchunks = (long)((n + chunk - 1) / chunk);
    double total = 0.0;
    if (n < (1u << 18)) {
        for (long c = 0; c < nchunks; ++c) total += chunk_sum(x + c * chunk, std::min(chunk, n - c * chunk));
        return total;
    }
    // Dynamic scheduling: the GB10 mixes fast and slow cores, so equal shares would wait on the slow ones.
    [[maybe_unused]] const long grain = std::max(1L, nchunks / (8L * cpu_threads()));
    #pragma omp parallel for schedule(dynamic, grain) reduction(+ : total)
    for (long c = 0; c < nchunks; ++c) total += chunk_sum(x + c * chunk, std::min(chunk, n - c * chunk));
    return total;
}

static float hash01(uint64_t i) {  // reproducible uniform [0, 1), independent of thread count
    uint64_t z = i + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

static void cpu_fill(float* x, size_t n, uint64_t seed) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i) x[i] = hash01((uint64_t)i ^ seed);
}

static void cpu_zero(float* x, size_t n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i) x[i] = 0.0f;
}

static float* host_alloc(size_t elems) {
    const size_t align = 2u << 20;  // 2 MB, so the kernel can back it with huge pages
    const size_t bytes = (elems * sizeof(float) + align - 1) / align * align;
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
#ifdef __linux__
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return (float*)p;
}

// ============================================================================
// 3. GPU kernels (compiled only by nvcc; stubs otherwise)
// ============================================================================

enum { SUM_BLOCKS = 4096, SUM_THREADS = 256 };

#ifdef __CUDACC__

// Each thread strides over the array, then the block reduces in shared memory
// and writes one partial. The host adds up the (at most 4096) partials.
__global__ void sum_kernel(const float* __restrict__ x, size_t n, float* __restrict__ partial) {
    __shared__ float sh[SUM_THREADS];
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) acc += x[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sh[0];
}

struct Gpu {
    bool ok = false;
    std::string name;
    cublasHandle_t blas = nullptr;
    float *A = nullptr, *B = nullptr, *C = nullptr, *partial = nullptr;
    int blocks = 0;  // partials pending from the last sum launch
    float host_partial[SUM_BLOCKS];
};

#define CU(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) die("CUDA: %s (%s, line %d)", cudaGetErrorString(e_), #call, __LINE__); \
    } while (0)

static bool gpu_init(Gpu& g, std::string& why) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        why = e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device";
        return false;
    }
    cudaDeviceProp prop;
    CU(cudaSetDevice(0));
    CU(cudaGetDeviceProperties(&prop, 0));
    g.name = prop.name;
    if (cublasCreate(&g.blas) != CUBLAS_STATUS_SUCCESS) {
        why = "cublasCreate failed";
        return false;
    }
    CU(cudaMalloc(&g.partial, SUM_BLOCKS * sizeof(float)));
    return g.ok = true;
}

static void gpu_free(Gpu& g) {
    if (g.A) cudaFree(g.A);
    if (g.B) cudaFree(g.B);
    if (g.C) cudaFree(g.C);
    g.A = g.B = g.C = nullptr;
}

static bool gpu_alloc(Gpu& g, size_t elems) {
    const size_t bytes = elems * sizeof(float);
    if (cudaMalloc(&g.A, bytes) == cudaSuccess && cudaMalloc(&g.B, bytes) == cudaSuccess &&
        cudaMalloc(&g.C, bytes) == cudaSuccess)
        return true;
    cudaGetLastError();  // clear the sticky error before the caller retries smaller
    gpu_free(g);
    return false;
}

static void gpu_upload(Gpu& g, const float* A, const float* B, size_t elems) {
    CU(cudaMemcpy(g.A, A, elems * sizeof(float), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(g.B, B, elems * sizeof(float), cudaMemcpyHostToDevice));
}

static void gpu_zero_C(Gpu& g, size_t elems) {
    CU(cudaMemset(g.C, 0, elems * sizeof(float)));
    CU(cudaDeviceSynchronize());  // finish now, so the clear never lands inside a timed call
}

static void gpu_wait(Gpu&) { CU(cudaDeviceSynchronize()); }

// C[:, 0:nc] = A * B[:, 0:nc] on the device. Returns once the work is queued.
static void gpu_gemm(Gpu& g, int N, int nc) {
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t s = cublasSgemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, nc, N, &one, g.A, N, g.B, N, &zero, g.C, N);
    if (s != CUBLAS_STATUS_SUCCESS) die("cublasSgemm failed (status %d, N=%d)", (int)s, N);
}

// Queue a sum over the first n elements of device A (of_C = false) or C.
static void gpu_sum_launch(Gpu& g, bool of_C, size_t n) {
    g.blocks = (int)std::min<size_t>(SUM_BLOCKS, (n + SUM_THREADS - 1) / SUM_THREADS);
    sum_kernel<<<g.blocks, SUM_THREADS>>>(of_C ? g.C : g.A, n, g.partial);
    CU(cudaGetLastError());
}

// Wait for the queued sum and return it.
static double gpu_sum_collect(Gpu& g) {
    CU(cudaMemcpy(g.host_partial, g.partial, g.blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < g.blocks; ++i) s += g.host_partial[i];
    return s;
}

static void gpu_mem_info(double& free_gb, double& total_gb) {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) { cudaGetLastError(); f = t = 0; }
    free_gb = f / 1073741824.0;
    total_gb = t / 1073741824.0;
}

#else  // built without nvcc: the GPU path is absent

struct Gpu { bool ok = false; std::string name; };
static bool gpu_init(Gpu&, std::string& why) { why = "built without nvcc"; return false; }
static void gpu_free(Gpu&) {}
static bool gpu_alloc(Gpu&, size_t) { return true; }
static void gpu_upload(Gpu&, const float*, const float*, size_t) {}
static void gpu_zero_C(Gpu&, size_t) {}
static void gpu_wait(Gpu&) {}
static void gpu_gemm(Gpu&, int, int) {}
static void gpu_sum_launch(Gpu&, bool, size_t) {}
static double gpu_sum_collect(Gpu&) { return 0.0; }
static void gpu_mem_info(double& f, double& t) { f = t = 0.0; }

#endif

// ============================================================================
// 4. The sweep: workloads x devices x sizes -> CSV
// ============================================================================

enum Workload { MATMUL = 0, SUM = 1, PIPE = 2, NUM_W = 3 };
enum Device { CPU = 0, GPU = 1, BOTH = 2, NUM_D = 3 };
static const char* const W_NAME[NUM_W] = {"matmul", "sum", "matmul+sum"};
static const char* const D_NAME[NUM_D] = {"cpu", "gpu", "cpu+gpu"};

struct Options {
    int min_n = 4, max_n = 32768, ppo = 2;
    double mem_gb = 90.0, budget_s = 600.0;
    bool dev[NUM_D] = {true, true, true};
    bool wl[NUM_W] = {true, true, true};
    std::string out = "results.csv";
    bool plot = true;
};

struct Ctx {
    float *A = nullptr, *B = nullptr, *C = nullptr;  // host, column-major
    Gpu gpu;
};

static double work_flops(int w, double N) {
    return w == MATMUL ? 2.0 * N * N * N : w == SUM ? N * N : 2.0 * N * N * N + N * N;
}

// One execution of workload w at size N. The job is U units long (columns for
// the matmul workloads, elements for the sum); the GPU takes units [0, ug) and
// the CPU takes [ug, U). The GPU share is queued first and runs while the CPU
// works on its own share. Returns the workload's scalar (0 for plain matmul).
static double run_once(Ctx& c, int w, int N, size_t ug) {
    const size_t U = (w == SUM) ? (size_t)N * N : (size_t)N;
    double r = 0.0;
    if (ug > 0) {
        if (w != SUM) gpu_gemm(c.gpu, N, (int)ug);
        if (w == SUM) gpu_sum_launch(c.gpu, false, ug);
        if (w == PIPE) gpu_sum_launch(c.gpu, true, ug * N);
    }
    if (ug < U) {
        if (w != SUM) cpu_gemm(N, (int)(U - ug), c.A, c.B + ug * N, c.C + ug * N);
        if (w == SUM) r += cpu_sum(c.A + ug, U - ug);
        if (w == PIPE) r += cpu_sum(c.C + ug * N, (U - ug) * N);
    }
    if (ug > 0) {
        if (w == MATMUL) gpu_wait(c.gpu);
        else r += gpu_sum_collect(c.gpu);
    }
    return r;
}

// Untimed check of a matmul result that is split at column ug.
static double checksum_C(Ctx& c, int N, size_t ug) {
    double s = 0.0;
    if (ug > 0) { gpu_sum_launch(c.gpu, true, ug * N); s += gpu_sum_collect(c.gpu); }
    if (ug < (size_t)N) s += cpu_sum(c.C + ug * N, ((size_t)N - ug) * N);
    return s;
}

static double mem_available_gb() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    double kb;
    while (f >> key >> kb) {
        if (key == "MemAvailable:") return kb / 1048576.0;
        f.ignore(256, '\n');
    }
    return -1.0;
}

static std::vector<int> size_grid(int min_n, int max_n, int ppo) {
    std::vector<int> g;
    for (int k = 0;; ++k) {
        const int n = (int)std::lround(min_n * std::pow(2.0, (double)k / ppo));
        if (n > max_n) break;
        if (g.empty() || n > g.back()) g.push_back(n);
    }
    if (g.empty() || g.back() < max_n) g.push_back(max_n);
    return g;
}

static int plot_csv(const std::string& csv_path, const std::string& svg_path);

static int run_sweep(const Options& o) {
    Ctx c;
    bool dev[NUM_D] = {o.dev[CPU], o.dev[GPU], o.dev[BOTH]};

    // ---- devices -----------------------------------------------------------
    if (dev[GPU] || dev[BOTH]) {
        std::string why;
        if (!gpu_init(c.gpu, why)) {
            std::printf("GPU unavailable (%s): running the CPU series only.\n", why.c_str());
            dev[GPU] = dev[BOTH] = false;
        }
    }
    if (dev[BOTH] && !(dev[CPU] && dev[GPU])) die("the cpu+gpu series needs both cpu and gpu in --devices");
    if (!dev[CPU] && !dev[GPU]) die("nothing to run");
    const bool use_gpu = c.gpu.ok && (dev[GPU] || dev[BOTH]);

    // ---- memory plan -------------------------------------------------------
    std::vector<int> grid = size_grid(o.min_n, o.max_n, o.ppo);
    const double avail = mem_available_gb();
    const double budget_gb = avail > 0 ? std::min(o.mem_gb, 0.85 * avail) : o.mem_gb;
    const int matrices = use_gpu ? 6 : 3;
    auto need_gb = [&](int n) { return matrices * (double)n * n * 4.0 / 1073741824.0 + 0.25; };
    while (grid.size() > 1 && need_gb(grid.back()) > budget_gb) {
        std::printf("N = %d needs %.1f GiB, over the %.1f GiB budget: dropped.\n", grid.back(),
                    need_gb(grid.back()), budget_gb);
        grid.pop_back();
    }
    if (need_gb(grid.back()) > budget_gb) die("even N = %d does not fit in %.1f GiB", grid.back(), budget_gb);

    // ---- allocate once at the largest size; step down if the system refuses -
    for (;;) {
        const size_t elems = (size_t)grid.back() * grid.back();
        bool ok = !use_gpu || gpu_alloc(c.gpu, elems);  // device first: it is the pickier allocator
        if (ok) {
            c.A = host_alloc(elems);
            c.B = host_alloc(elems);
            c.C = host_alloc(elems);
            ok = c.A && c.B && c.C;
        }
        if (ok) break;
        std::printf("Allocation for N = %d failed; trying the next size down.\n"
                    "(On the GB10, page cache can block CUDA allocations. To free it:\n"
                    "   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches')\n", grid.back());
        gpu_free(c.gpu);
        std::free(c.A); std::free(c.B); std::free(c.C);
        c.A = c.B = c.C = nullptr;
        if (grid.size() == 1) die("could not allocate even the smallest size");
        grid.pop_back();
    }
    const int n_max = grid.back();
    const size_t elems_max = (size_t)n_max * n_max;

    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    char date[64];
    const std::time_t tt = std::time(nullptr);
    std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", std::localtime(&tt));
    double gfree = 0, gtotal = 0;
    if (use_gpu) gpu_mem_info(gfree, gtotal);

    std::printf("host         %s\n", host);
    std::printf("cpu          %d threads, matmul via %s\n", cpu_threads(), cpu_gemm_backend().c_str());
    if (blas_has_own_threads())
        std::printf("             note: this OpenBLAS uses its own pthread pool, which competes with the\n"
                    "             OpenMP threads of the sum. For cleaner numbers install the OpenMP\n"
                    "             build (sudo apt install libopenblas-openmp-dev) and run make again.\n");
    if (use_gpu) std::printf("gpu          %s (CUDA reports %.1f of %.1f GiB free)\n", c.gpu.name.c_str(), gfree, gtotal);
    std::printf("sizes        N = %d .. %d (%s .. %s elements), %zu points\n", grid.front(), n_max,
                fmt_count((double)grid.front() * grid.front()).c_str(), fmt_count((double)elems_max).c_str(),
                grid.size());
    std::printf("memory       %.2f GiB allocated (%d matrices of %.2f GiB); budget %.1f GiB",
                matrices * elems_max * 4.0 / 1073741824.0, matrices, elems_max * 4.0 / 1073741824.0, budget_gb);
    if (avail > 0) std::printf(", %.1f GiB available", avail);
    std::printf("\nbudget       a point predicted to take over %s ends its series\n\n", fmt_time(o.budget_s).c_str());

    cpu_fill(c.A, elems_max, 0x1111);
    cpu_fill(c.B, elems_max, 0x2222ull << 32);
    cpu_zero(c.C, elems_max);
    if (use_gpu) {
        gpu_upload(c.gpu, c.A, c.B, elems_max);
        gpu_zero_C(c.gpu, elems_max);
    }

    // ---- CSV ---------------------------------------------------------------
    std::FILE* csv = std::fopen(o.out.c_str(), "w");
    if (!csv) die("cannot write %s", o.out.c_str());
    std::fprintf(csv, "# host: %s\n# date: %s\n# cpu_threads: %d\n# cpu_matmul: %s\n# gpu: %s\n", host, date,
                 cpu_threads(), cpu_gemm_backend().c_str(), use_gpu ? c.gpu.name.c_str() : "none");
    std::fprintf(csv, "workload,device,n,elements,flops,seconds,gflops,reps,gpu_share,checksum\n");
    std::fflush(csv);

    // ---- the sweep ---------------------------------------------------------
    struct SeriesState { bool stopped = false; int last_n = 0; double last_t = 0, rate = 0; };
    SeriesState st[NUM_W][NUM_D];
    std::printf("%-11s %-8s %7s %10s %12s %12s %10s  %s\n", "workload", "device", "N", "elements", "time",
                "GFLOP/s", "reps", "notes");
    const double t_start = now_s();
    int warnings = 0;

    for (int N : grid) {
        const size_t n2 = (size_t)N * N;
        for (int w = 0; w < NUM_W; ++w) {
            if (!o.wl[w]) continue;
            const double work = work_flops(w, N);
            const size_t U = (w == SUM) ? n2 : (size_t)N;
            double reference = NAN;  // first checksum at this (workload, N)
            for (int d = 0; d < NUM_D; ++d) {
                SeriesState& s = st[w][d];
                if (!dev[d] || s.stopped) continue;

                // Units given to the GPU.
                size_t ug = (d == GPU) ? U : 0;
                double predicted = s.last_n ? s.last_t * work / work_flops(w, s.last_n) : 0.0;
                if (d == BOTH) {
                    const double rc = st[w][CPU].rate, rg = st[w][GPU].rate;
                    if (rc <= 0 || rg <= 0) continue;  // a solo series has no measurement yet
                    const long long want = std::llround(rg / (rc + rg) * (double)U);
                    ug = (size_t)std::min<long long>(std::max<long long>(want, 1), (long long)U - 1);
                    if (!s.last_n) predicted = work / (rc + rg);
                }
                if (predicted > o.budget_s) {
                    s.stopped = true;
                    std::printf("%-11s %-8s %7d %10s  series ends here: this point is predicted to take %s (budget %s)\n",
                                W_NAME[w], D_NAME[d], N, fmt_count((double)n2).c_str(), fmt_time(predicted).c_str(),
                                fmt_time(o.budget_s).c_str());
                    continue;
                }

                // Clear the output so the checksum below reflects this run only.
                if (w != SUM) {
                    if (ug < U) cpu_zero(c.C, n2);
                    if (ug > 0) gpu_zero_C(c.gpu, n2);
                }

                double result = 0.0;
                const Timing t = measure([&] { result = run_once(c, w, N, ug); });
                const double checksum = (w == MATMUL) ? checksum_C(c, N, ug) : result;

                s.last_n = N;
                s.last_t = t.sec;
                s.rate = work / t.sec;
                const double share = (d == BOTH) ? (double)ug / (double)U : (d == GPU ? 1.0 : 0.0);

                std::string note;
                if (d == BOTH) {
                    char b[64];
                    std::snprintf(b, sizeof b, "gpu share %.0f%%", 100.0 * share);
                    note = b;
                }
                if (std::isnan(reference)) reference = checksum;
                else if (std::fabs(checksum - reference) > 1e-3 * std::fabs(reference)) {
                    note += note.empty() ? "" : ", ";
                    note += "WARNING: checksum differs from the first device";
                    ++warnings;
                }
                std::printf("%-11s %-8s %7d %10s %12s %12.3f %10ld  %s\n", W_NAME[w], D_NAME[d], N,
                            fmt_count((double)n2).c_str(), fmt_time(t.sec).c_str(), s.rate / 1e9, t.reps,
                            note.c_str());
                std::fflush(stdout);
                std::fprintf(csv, "%s,%s,%d,%zu,%.6e,%.6e,%.6f,%ld,%.4f,%.9e\n", W_NAME[w], D_NAME[d], N, n2, work,
                             t.sec, s.rate / 1e9, t.reps, share, checksum);
                std::fflush(csv);
            }
        }
    }
    std::fclose(csv);
    gpu_free(c.gpu);
    std::free(c.A); std::free(c.B); std::free(c.C);

    std::printf("\nDone in %.0f s. Results: %s\n", now_s() - t_start, o.out.c_str());
    if (warnings) std::printf("%d checksum warning(s): see the notes column.\n", warnings);
    if (o.plot) {
        std::string svg = o.out;
        const size_t dot = svg.rfind('.');
        if (dot != std::string::npos && dot > svg.find_last_of('/') + 1) svg.erase(dot);
        return plot_csv(o.out, svg + ".svg");
    }
    return 0;
}

// ============================================================================
// 5. Plotting: CSV -> SVG. Independent of everything above except fmt_*.
//    Open the .svg in any browser; hover a point for its exact values.
// ============================================================================

namespace plot {

struct Row { std::string w, d; double n = 0, elems = 0, sec = 0, gflops = 0, share = 0; };
struct Point { double x, y; std::string tip; };
struct Curve { int dev; std::vector<Point> pts; };
enum YKind { Y_TIME, Y_RATE, Y_SPEEDUP };
struct Panel { std::string title; YKind kind; std::vector<Curve> curves; };

static const char* const DEV[3] = {"cpu", "gpu", "cpu+gpu"};
static const char* const DEV_LABEL[3] = {"CPU", "GPU", "CPU+GPU"};
static const char* const LIGHT[3] = {"#2a78d6", "#eb6834", "#1baf7a"};
static const char* const DARK[3] = {"#3987e5", "#d95926", "#199e70"};

static std::string esc(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '&') o += "&amp;";
        else if (ch == '<') o += "&lt;";
        else if (ch == '>') o += "&gt;";
        else if (ch == '"') o += "&quot;";
        else o += ch;
    }
    return o;
}

static std::string num(double v, const char* fmt = "%.4g") {
    char b[48];
    std::snprintf(b, sizeof b, fmt, v);
    return b;
}

static std::string px(double v) { return num(v, "%.1f"); }

static std::string tick_label(double v, YKind kind) {
    if (kind == Y_TIME) return fmt_time(v, true);
    return fmt_count(v) + (kind == Y_SPEEDUP ? "\xC3\x97" : "");
}

struct Tick { double v; bool labelled; };

// Decade ticks inside [lo, hi]; with fewer than three decades, 2x and 5x too.
static std::vector<Tick> log_ticks(double lo, double hi, int max_labels) {
    const int e0 = (int)std::floor(std::log10(lo)), e1 = (int)std::ceil(std::log10(hi));
    int decades = 0;
    for (int e = e0; e <= e1; ++e) {
        const double v = std::pow(10.0, e);
        if (v >= lo * 0.999 && v <= hi * 1.001) ++decades;
    }
    const int every = std::max(1, (decades + max_labels - 1) / max_labels);
    std::vector<Tick> t;
    for (int e = e0; e <= e1; ++e) {
        for (int m : {1, 2, 5}) {
            if (m != 1 && decades >= 3) continue;
            const double v = m * std::pow(10.0, e);
            if (v < lo * 0.999 || v > hi * 1.001) continue;
            t.push_back({v, m != 1 || ((e % every) + every) % every == 0});
        }
    }
    return t;
}

static bool load(const std::string& path, std::vector<Row>& rows, std::map<std::string, std::string>& meta) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::vector<std::string> head;
    auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string cell;
        while (std::getline(ss, cell, ',')) out.push_back(cell);
        return out;
    };
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t colon = line.find(':');
            if (colon != std::string::npos && colon + 2 <= line.size())
                meta[line.substr(2, colon - 2)] = line.substr(std::min(line.size(), colon + 2));
            continue;
        }
        const std::vector<std::string> cells = split(line);
        if (head.empty()) { head = cells; continue; }
        Row r;
        for (size_t i = 0; i < cells.size() && i < head.size(); ++i) {
            const std::string& h = head[i];
            if (h == "workload") r.w = cells[i];
            else if (h == "device") r.d = cells[i];
            else if (h == "n") r.n = std::atof(cells[i].c_str());
            else if (h == "elements") r.elems = std::atof(cells[i].c_str());
            else if (h == "seconds") r.sec = std::atof(cells[i].c_str());
            else if (h == "gflops") r.gflops = std::atof(cells[i].c_str());
            else if (h == "gpu_share") r.share = std::atof(cells[i].c_str());
        }
        if (r.elems > 0 && r.sec > 0 && r.gflops > 0) rows.push_back(r);
    }
    return true;
}

static int render(const std::string& csv_path, const std::string& svg_path) {
    std::vector<Row> rows;
    std::map<std::string, std::string> meta;
    if (!load(csv_path, rows, meta)) { std::fprintf(stderr, "error: cannot read %s\n", csv_path.c_str()); return 1; }
    if (rows.empty()) { std::fprintf(stderr, "error: no data rows in %s\n", csv_path.c_str()); return 1; }

    // ---- panels: one row per workload; time, throughput, speedup over CPU ---
    const char* const WL[3] = {"matmul", "sum", "matmul+sum"};
    const char* const WL_LABEL[3] = {"Matmul  C = A\xC2\xB7" "B", "Sum  \xCE\xA3x", "Matmul + sum  \xCE\xA3(A\xC2\xB7" "B)"};
    std::vector<std::vector<Panel>> grid;
    bool present[3] = {false, false, false};
    double xmin = 1e300, xmax = 0;
    for (int w = 0; w < 3; ++w) {
        Panel time{std::string(WL_LABEL[w]) + ": time per call", Y_TIME, {}};
        Panel rate{std::string(WL_LABEL[w]) + ": throughput, GFLOP/s", Y_RATE, {}};
        Panel gain{std::string(WL_LABEL[w]) + ": speedup over CPU", Y_SPEEDUP, {}};
        std::map<double, double> cpu_sec;  // elements -> CPU seconds
        for (const Row& r : rows)
            if (r.w == WL[w] && r.d == "cpu") cpu_sec[r.elems] = r.sec;
        bool any = false;
        for (int d = 0; d < 3; ++d) {
            Curve ct{d, {}}, cr{d, {}}, cg{d, {}};
            for (const Row& r : rows) {
                if (r.w != WL[w] || r.d != DEV[d]) continue;
                std::string tip = std::string(DEV_LABEL[d]) + " \xC2\xB7 N = " + num(r.n, "%.0f") + " (" +
                                  fmt_count(r.elems) + " elements) \xC2\xB7 " + fmt_time(r.sec, true) + " \xC2\xB7 " +
                                  num(r.gflops, "%.3g") + " GFLOP/s";
                if (d == 2) tip += " \xC2\xB7 GPU share " + num(100.0 * r.share, "%.0f") + "%";
                ct.pts.push_back({r.elems, r.sec, tip});
                cr.pts.push_back({r.elems, r.gflops, tip});
                const auto base = cpu_sec.find(r.elems);
                if (d != 0 && base != cpu_sec.end()) {
                    const double x = base->second / r.sec;
                    cg.pts.push_back({r.elems, x, tip + " \xC2\xB7 " + num(x, "%.3g") + "\xC3\x97 the CPU"});
                }
                xmin = std::min(xmin, r.elems);
                xmax = std::max(xmax, r.elems);
            }
            auto by_x = [](const Point& a, const Point& b) { return a.x < b.x; };
            for (Curve* cv : {&ct, &cr, &cg}) std::sort(cv->pts.begin(), cv->pts.end(), by_x);
            if (!ct.pts.empty()) { time.curves.push_back(ct); rate.curves.push_back(cr); present[d] = any = true; }
            if (!cg.pts.empty()) gain.curves.push_back(cg);
        }
        if (any) grid.push_back({time, rate, gain});
    }
    if (grid.empty()) { std::fprintf(stderr, "error: no known workloads in %s\n", csv_path.c_str()); return 1; }
    if (xmax <= xmin) xmax = xmin * 10;

    // ---- geometry ----------------------------------------------------------
    const double W = 1500, pad_l = 28, pad_r = 20, gap_x = 22, head_h = 108;
    const double col_w = (W - pad_l - pad_r - 2 * gap_x) / 3, gutter = 58, end_pad = 70;
    const double plot_w = col_w - gutter - end_pad, title_h = 28, plot_h = 220, row_h = title_h + plot_h + 56;
    const double H = head_h + grid.size() * row_h + 64;
    const double lx0 = std::log10(xmin / 1.6), lx1 = std::log10(xmax * 1.6);

    std::ostringstream s;
    s << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << W << " " << H << "\" width=\"" << W
      << "\" height=\"" << H << "\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">\n";
    // Light colours are inline attributes; the stylesheet only restyles for dark mode.
    s << "<style>\n@media (prefers-color-scheme: dark) {\n"
         "  .bg{fill:#1a1a19} .ink{fill:#f0efec} .ink2{fill:#c3c2b7} .grid{stroke:#2c2c2a} .axis{stroke:#383835}\n"
         "  .ring{stroke:#1a1a19}\n";
    for (int d = 0; d < 3; ++d)
        s << "  .l" << d << "{stroke:" << DARK[d] << "} .f" << d << "{fill:" << DARK[d] << "}\n";
    s << "}\n.tick{font-size:11px;font-variant-numeric:tabular-nums}\n.hit:hover + circle{r:5.5px}\n</style>\n";
    s << "<rect class=\"bg\" width=\"" << W << "\" height=\"" << H << "\" fill=\"#fcfcfb\"/>\n";

    // ---- header: title, run details, legend ---------------------------------
    std::string sub;
    auto add = [&](const std::string& part) { if (!part.empty()) sub += (sub.empty() ? "" : "  \xC2\xB7  ") + part; };
    if (meta.count("host")) add(meta["host"]);
    if (meta.count("gpu") && meta["gpu"] != "none") add("GPU: " + meta["gpu"]);
    if (meta.count("cpu_threads")) add(meta["cpu_threads"] + " CPU threads");
    if (meta.count("cpu_matmul")) {
        std::string m = meta["cpu_matmul"];
        add("CPU matmul: " + m);
    }
    add("float32");
    if (meta.count("date")) add(meta["date"]);
    s << "<text class=\"ink\" x=\"" << pad_l << "\" y=\"36\" font-size=\"20\" font-weight=\"600\" fill=\"#0b0b0b\">"
         "Scaling sweep: CPU, GPU and CPU+GPU</text>\n";
    s << "<text class=\"ink2\" x=\"" << pad_l << "\" y=\"58\" font-size=\"12.5\" fill=\"#52514e\">" << esc(sub)
      << "</text>\n";
    double lx = pad_l;
    for (int d = 0; d < 3; ++d) {
        if (!present[d]) continue;
        s << "<line class=\"l" << d << "\" x1=\"" << lx << "\" x2=\"" << lx + 26 << "\" y1=\"84\" y2=\"84\" stroke=\""
          << LIGHT[d] << "\" stroke-width=\"2\" stroke-linecap=\"round\"/>"
          << "<circle class=\"f" << d << "\" cx=\"" << lx + 13 << "\" cy=\"84\" r=\"4\" fill=\"" << LIGHT[d] << "\"/>"
          << "<text class=\"ink\" x=\"" << lx + 34 << "\" y=\"88.5\" font-size=\"13\" fill=\"#0b0b0b\">" << DEV_LABEL[d]
          << "</text>\n";
        lx += 34 + 8.2 * std::strlen(DEV_LABEL[d]) + 30;
    }

    // ---- panels ------------------------------------------------------------
    for (size_t r = 0; r < grid.size(); ++r) {
        for (int col = 0; col < 3; ++col) {
            const Panel& p = grid[r][col];
            const double x0 = pad_l + col * (col_w + gap_x) + gutter, y0 = head_h + r * row_h + title_h;
            s << "<text class=\"ink\" x=\"" << px(x0 - gutter) << "\" y=\"" << px(y0 - 12)
              << "\" font-size=\"13\" font-weight=\"600\" fill=\"#0b0b0b\">" << esc(p.title) << "</text>\n";

            double ymin = 1e300, ymax = 0;
            for (const Curve& c : p.curves)
                for (const Point& pt : c.pts) { ymin = std::min(ymin, pt.y); ymax = std::max(ymax, pt.y); }
            if (p.kind == Y_SPEEDUP && ymax > 0) { ymin = std::min(ymin, 1.0); ymax = std::max(ymax, 1.0); }
            if (ymax <= 0) {
                s << "<text class=\"ink2\" x=\"" << px(x0 + plot_w / 2) << "\" y=\"" << px(y0 + plot_h / 2)
                  << "\" font-size=\"12\" text-anchor=\"middle\" fill=\"#52514e\">needs a CPU series and one other</text>\n";
                continue;
            }
            const double ly0 = std::log10(ymin / 1.8), ly1 = std::log10(ymax * 1.8);
            auto X = [&](double v) { return x0 + (std::log10(v) - lx0) / (lx1 - lx0) * plot_w; };
            auto Y = [&](double v) { return y0 + plot_h - (std::log10(v) - ly0) / (ly1 - ly0) * plot_h; };

            // grid and tick labels
            for (const Tick& t : log_ticks(std::pow(10, ly0), std::pow(10, ly1), 8)) {
                const bool one = p.kind == Y_SPEEDUP && std::fabs(t.v - 1.0) < 1e-9;
                s << "<line class=\"" << (one ? "axis" : "grid") << "\" x1=\"" << px(x0) << "\" x2=\"" << px(x0 + plot_w)
                  << "\" y1=\"" << px(Y(t.v)) << "\" y2=\"" << px(Y(t.v)) << "\" stroke=\"" << (one ? "#c3c2b7" : "#e1e0d9")
                  << "\" stroke-width=\"" << (one ? 1.5 : 1) << "\"/>";
                if (t.labelled)
                    s << "<text class=\"tick\" x=\"" << px(x0 - 8) << "\" y=\"" << px(Y(t.v) + 4)
                      << "\" text-anchor=\"end\" fill=\"#898781\">" << tick_label(t.v, p.kind) << "</text>";
                s << "\n";
            }
            s << "<line class=\"axis\" x1=\"" << px(x0) << "\" x2=\"" << px(x0 + plot_w) << "\" y1=\"" << px(y0 + plot_h)
              << "\" y2=\"" << px(y0 + plot_h) << "\" stroke=\"#c3c2b7\"/>\n";
            for (const Tick& t : log_ticks(std::pow(10, lx0), std::pow(10, lx1), 9)) {
                if (!t.labelled) continue;
                s << "<line class=\"axis\" x1=\"" << px(X(t.v)) << "\" x2=\"" << px(X(t.v)) << "\" y1=\"" << px(y0 + plot_h)
                  << "\" y2=\"" << px(y0 + plot_h + 4) << "\" stroke=\"#c3c2b7\"/>"
                  << "<text class=\"tick\" x=\"" << px(X(t.v)) << "\" y=\"" << px(y0 + plot_h + 18)
                  << "\" text-anchor=\"middle\" fill=\"#898781\">" << fmt_count(t.v) << "</text>\n";
            }
            if (r + 1 == grid.size())
                s << "<text class=\"ink2\" x=\"" << px(x0 + plot_w / 2) << "\" y=\"" << px(y0 + plot_h + 40)
                  << "\" font-size=\"12\" text-anchor=\"middle\" fill=\"#52514e\">elements (N \xC3\x97 N)</text>\n";

            // curves, then markers on top
            for (const Curve& c : p.curves) {
                s << "<polyline class=\"l" << c.dev << "\" fill=\"none\" stroke=\"" << LIGHT[c.dev]
                  << "\" stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"";
                for (const Point& pt : c.pts) s << px(X(pt.x)) << "," << px(Y(pt.y)) << " ";
                s << "\"/>\n";
            }
            for (const Curve& c : p.curves)
                for (const Point& pt : c.pts)
                    s << "<g><title>" << esc(pt.tip) << "</title><circle class=\"hit\" cx=\"" << px(X(pt.x)) << "\" cy=\""
                      << px(Y(pt.y)) << "\" r=\"11\" fill=\"transparent\"/><circle class=\"f" << c.dev << " ring\" cx=\""
                      << px(X(pt.x)) << "\" cy=\"" << px(Y(pt.y)) << "\" r=\"4\" fill=\"" << LIGHT[c.dev]
                      << "\" stroke=\"#fcfcfb\" stroke-width=\"1.5\" pointer-events=\"none\"/></g>\n";

            // direct labels at the line ends, only where they do not collide
            std::vector<std::pair<double, int>> ends;
            for (const Curve& c : p.curves) ends.push_back({Y(c.pts.back().y), (int)(&c - &p.curves[0])});
            std::sort(ends.begin(), ends.end());
            bool clear = ends.size() > 1 || p.kind == Y_SPEEDUP;
            for (size_t i = 1; i < ends.size(); ++i) clear = clear && ends[i].first - ends[i - 1].first >= 14;
            if (clear)
                for (const auto& e : ends) {
                    const Curve& c = p.curves[e.second];
                    s << "<text class=\"ink2\" x=\"" << px(X(c.pts.back().x) + 9) << "\" y=\"" << px(e.first + 4)
                      << "\" font-size=\"11.5\" fill=\"#52514e\">" << DEV_LABEL[c.dev] << "</text>\n";
                }
        }
    }

    // ---- footnotes ---------------------------------------------------------
    const double fy = head_h + grid.size() * row_h + 22;
    s << "<text class=\"ink2\" x=\"" << pad_l << "\" y=\"" << fy << "\" font-size=\"11.5\" fill=\"#52514e\">"
         "Throughput counts 2N\xC2\xB3 flops for the matmul and one add per element for the sum. "
         "GPU times are compute-only: the inputs are already on the device.</text>\n";
    s << "<text class=\"ink2\" x=\"" << pad_l << "\" y=\"" << fy + 17 << "\" font-size=\"11.5\" fill=\"#52514e\">"
         "CPU+GPU runs both at once, splitting each job in proportion to the two devices' measured solo rates. "
         "Both axes are logarithmic. Hover a point for its values.</text>\n";
    s << "</svg>\n";

    std::ofstream out(svg_path);
    if (!out) { std::fprintf(stderr, "error: cannot write %s\n", svg_path.c_str()); return 1; }
    out << s.str();
    std::printf("Plots: %s (open it in a browser)\n", svg_path.c_str());
    return 0;
}

}  // namespace plot

static int plot_csv(const std::string& csv_path, const std::string& svg_path) { return plot::render(csv_path, svg_path); }

// ============================================================================
// 6. Command line
// ============================================================================

static void usage() {
    std::printf(
        "usage:\n"
        "  gb10_sweep [run] [options]        run the sweep, write CSV and SVG\n"
        "  gb10_sweep plot <csv> [out.svg]   redraw the plots from an existing CSV\n\n"
        "options:\n"
        "  --min-n N        smallest matrix side (default 4, i.e. 16 elements)\n"
        "  --max-n N        largest matrix side (default 32768, i.e. 1.07e9 elements)\n"
        "  --ppo K          grid points per doubling of N (default 2)\n"
        "  --mem-gb G       memory budget for all buffers, in GiB (default 90)\n"
        "  --budget S       end a series when its next point is predicted over S seconds (default 600)\n"
        "  --devices LIST   any of cpu,gpu,both (default all three)\n"
        "  --workloads LIST any of matmul,sum,pipe (default all three; pipe = matmul+sum)\n"
        "  --out FILE       CSV path (default results.csv; the SVG takes the same stem)\n"
        "  --no-plot        skip the SVG\n"
        "  --quick          short smoke test: --max-n 1024 --budget 10\n\n"
        "CPU threads follow OMP_NUM_THREADS (and OPENBLAS_NUM_THREADS when OpenBLAS is linked).\n");
}

static bool has(const std::string& list, const char* item) {
    std::stringstream ss(list);
    std::string tok;
    while (std::getline(ss, tok, ',')) if (tok == item) return true;
    return false;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // keep progress visible through tee or a pipe
    std::vector<std::string> a(argv + 1, argv + argc);
    if (!a.empty() && a[0] == "plot") {
        if (a.size() < 2) { usage(); return 1; }
        std::string svg = a.size() > 2 ? a[2] : a[1];
        if (a.size() <= 2) {
            const size_t dot = svg.rfind('.');
            if (dot != std::string::npos && dot > svg.find_last_of('/') + 1) svg.erase(dot);
            svg += ".svg";
        }
        return plot_csv(a[1], svg);
    }

    Options o;
    for (size_t i = (!a.empty() && a[0] == "run") ? 1 : 0; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= a.size()) die("%s needs a value", k.c_str());
            return a[++i];
        };
        if (k == "--min-n") o.min_n = std::atoi(val().c_str());
        else if (k == "--max-n") o.max_n = std::atoi(val().c_str());
        else if (k == "--ppo") o.ppo = std::atoi(val().c_str());
        else if (k == "--mem-gb") o.mem_gb = std::atof(val().c_str());
        else if (k == "--budget") o.budget_s = std::atof(val().c_str());
        else if (k == "--out") o.out = val();
        else if (k == "--no-plot") o.plot = false;
        else if (k == "--quick") { o.max_n = 1024; o.budget_s = 10; }
        else if (k == "--devices") {
            const std::string v = val();
            o.dev[CPU] = has(v, "cpu"); o.dev[GPU] = has(v, "gpu"); o.dev[BOTH] = has(v, "both");
        } else if (k == "--workloads") {
            const std::string v = val();
            o.wl[MATMUL] = has(v, "matmul"); o.wl[SUM] = has(v, "sum"); o.wl[PIPE] = has(v, "pipe");
        } else if (k == "-h" || k == "--help") { usage(); return 0; }
        else { usage(); die("unknown option %s", k.c_str()); }
    }
    if (o.min_n < 2 || o.max_n < o.min_n || o.max_n > 131072 || o.ppo < 1 || o.ppo > 16) die("bad size options");
    if (o.mem_gb <= 0 || o.budget_s <= 0) die("bad --mem-gb or --budget");
    if (!(o.wl[MATMUL] || o.wl[SUM] || o.wl[PIPE])) die("no workloads selected");
    return run_sweep(o);
}
