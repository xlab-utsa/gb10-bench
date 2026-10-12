// gpu.cu - GPU backends for the gb10_stress suite: CUDA kernels for every
// workload and precision, with cuBLAS doing the matrix products where it has a
// routine for the format. Compiled by nvcc; with -DGPU_EMULATE a plain C++
// compiler builds the same kernels against cuda_emu.hpp to run them on the CPU.
//
// Every kernel is one-dimensional: thread t of n does the t-th piece of work,
// and no kernel uses shared memory or __syncthreads().
#include "common.hpp"
#include "trace.hpp"

#include <algorithm>
#include <climits>
#include <map>

#ifdef GPU_EMULATE
#include "cuda_emu.hpp"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif

enum { TPB = 256, SMCH = 256, FBC = 64, RED_T1 = 262144, RED_T2 = 1024, HIST_T = 65536 };

#define CU(call)                                                                        \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) fail("CUDA: %s (%s)", cudaGetErrorString(e_), #call);    \
    } while (0)

struct GpuState {
    bool ok = false;
    cublasHandle_t blas;
    double* red = nullptr;  // scratch for reductions
    bool use_h = false, use_i8 = false, use_sb = false;  // cuBLAS routines that passed the start-up probe
    std::string name;
    int sms = 0;
};
static GpuState G;

static void gsync() { CU(cudaDeviceSynchronize()); }

// ============================================================================
// Tracing support (compiled only into the tracer)
// ============================================================================
#ifdef GB10_TRACE
struct KtRec { long long c0, c1; unsigned sm, pad; };  // one per thread block: first start, last end, SM
#ifndef GPU_EMULATE
static __device__ __forceinline__ unsigned kt_smid() {
#ifdef TRACE_NO_SMID
    return 0;
#else
    unsigned r;
    asm("mov.u32 %0, %smid;" : "=r"(r));  // PTX special register: the SM this thread runs on
    return r;
#endif
}
#endif
// Constructed first thing in every kernel. One thread per warp reports for its block.
struct KtScope {
    KtRec* r;
    __device__ KtScope(KtRec* buf, unsigned cap) {
        r = (blockIdx.x < cap && (threadIdx.x & 31) == 0) ? buf + blockIdx.x : (KtRec*)0;
        if (r) {
            atomicMin(&r->c0, (long long)clock64());
            if (threadIdx.x == 0) r->sm = kt_smid();
        }
    }
    __device__ ~KtScope() { if (r) atomicMax(&r->c1, (long long)clock64()); }
};
#define KT_PARAM , KtRec* kt_buf, unsigned kt_cap
#define KT_SCOPE KtScope kt_scope(kt_buf, kt_cap)
#else
#define KT_PARAM
#define KT_SCOPE
#endif

static __device__ __forceinline__ long long gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }

#ifdef GPU_EMULATE
#define RAW_LAUNCH(n, kernel, ...) emu::launch((long long)(n), [&]() { kernel(__VA_ARGS__); })
#else
#define RAW_LAUNCH(n, kernel, ...)                                                                  \
    do {                                                                                            \
        kernel<<<(unsigned)(((long long)(n) + TPB - 1) / TPB), TPB>>>(__VA_ARGS__);                  \
        CU(cudaGetLastError());                                                                     \
    } while (0)
#endif

#ifdef GB10_TRACE
__global__ void k_kt_init(long long n, KtRec* b) {
    const long long t = gid();
    if (t >= n) return;
    b[t].c0 = LLONG_MAX; b[t].c1 = LLONG_MIN; b[t].sm = 0xffffffffu; b[t].pad = 0;
}

struct KAttr { int regs; size_t shared, local; std::string name; };
struct KtState {
    KtRec* buf = nullptr;
    unsigned cap = 1u << 20;      // blocks recorded per launch
    long detail = 512;            // launches with at most this many blocks also emit one event per block
    int clock_khz = 0;
    size_t io_r = 0, io_w = 0;    // estimated bytes the next launch reads / writes
    std::vector<KtRec> host;
    std::map<std::string, KAttr> attrs;  // one entry per kernel instantiation
};
static KtState g_kt;

// Attributes and display name of a kernel. Template kernels are written
// "k_name<P>" at the launch site; the storage type the instantiation works on
// is read off its signature, so the log says k_name<fp16> and the registers
// are those of that instantiation.
template <class F> static const KAttr& kt_attrs(const char* name, F f) {
    const std::string sig = __PRETTY_FUNCTION__, key = std::string(name) + "|" + sig;
    auto it = g_kt.attrs.find(key);
    if (it != g_kt.attrs.end()) return it->second;
    KAttr k = {-1, 0, 0, name};
    const size_t lt = k.name.find("<P>");
    if (lt != std::string::npos) {
        size_t args = sig.find("(*)(");
        if (args == std::string::npos) args = sig.find("F = ");
        const size_t star = args == std::string::npos ? std::string::npos : sig.find('*', args + 4);
        const std::string first = star == std::string::npos ? "" : sig.substr(args, star - args);
        const char* type = first.find("half") != std::string::npos || first.find("Float16") != std::string::npos ? "fp16"
                         : first.find("unsigned char") != std::string::npos ? "int4"
                         : first.find("signed char") != std::string::npos ? "int8"
                         : first.find("double") != std::string::npos ? "f64"
                         : first.find("float") != std::string::npos ? "fp32"
                         : first.find("int") != std::string::npos ? "int32" : "?";
        k.name.replace(lt, 3, std::string("<") + type + ">");
    }
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, f) == cudaSuccess) { k.regs = a.numRegs; k.shared = a.sharedSizeBytes; k.local = a.localSizeBytes; }
    else cudaGetLastError();
    return g_kt.attrs[key] = k;
}

static void tr_gpu_io(size_t r, size_t w) { g_kt.io_r = r; g_kt.io_w = w; }

struct KtLaunch {
    const char* name; long long n; const KAttr& a; uint64_t t0; unsigned blocks, rec;
    KtLaunch(long long n_, const KAttr& a_) : name(a_.name.c_str()), n(n_), a(a_) {
        blocks = (unsigned)((n + TPB - 1) / TPB);
        rec = std::min(blocks, g_kt.cap);
        RAW_LAUNCH(rec, k_kt_init, (long long)rec, g_kt.buf);
        gsync();
        t0 = tr::now_ns();
    }
    void done() {
        gsync();
        const uint64_t t1 = tr::now_ns();
        g_kt.host.resize(rec);
        CU(cudaMemcpy(g_kt.host.data(), g_kt.buf, rec * sizeof(KtRec), cudaMemcpyDeviceToHost));
        long long cmin = LLONG_MAX, cmax = LLONG_MIN;
        for (const KtRec& r : g_kt.host) if (r.c1 >= r.c0) { cmin = std::min(cmin, r.c0); cmax = std::max(cmax, r.c1); }
        const bool have = cmax >= cmin;
        const double ns_per_cycle = g_kt.clock_khz > 0 ? 1e6 / g_kt.clock_khz : 1.0;
        // Per SM: the blocks it ran, how many were resident at once (exact, from the
        // recorded intervals), and the average number resident in each of NB slices
        // of the kernel's duration.
        enum { NB = 32 };
        struct Sm { std::vector<std::pair<long long, long long>> iv; long long c0 = LLONG_MAX, c1 = LLONG_MIN, busy = 0; };
        std::map<unsigned, Sm> sms;
        const double span = have ? (double)(cmax - cmin) + 1.0 : 1.0, bw = span / NB;
        for (const KtRec& r : g_kt.host) {
            if (r.c1 < r.c0) continue;
            Sm& s = sms[r.sm];
            s.iv.push_back({r.c0, r.c1});
            s.c0 = std::min(s.c0, r.c0); s.c1 = std::max(s.c1, r.c1); s.busy += r.c1 - r.c0;
        }
        TR_EVENT("gpu.kernel", t0, t1 - t0,
                 "\"kernel\":\"%s\",\"threads\":%lld,\"blocks\":%u,\"block_size\":%d,\"regs_per_thread\":%d,"
                 "\"shared_bytes\":%zu,\"local_bytes_per_thread\":%zu,\"sms_used\":%zu,\"blocks_recorded\":%u,"
                 "\"device_cycles\":%lld,\"device_ns_est\":%.0f,\"bytes_read_est\":%zu,\"bytes_written_est\":%zu,"
                 "\"launch_tid\":%d,\"launch_cpu\":%d",
                 name, n, blocks, (int)TPB, a.regs, a.shared, a.local, sms.size(), rec, have ? cmax - cmin : 0LL,
                 have ? (cmax - cmin) * ns_per_cycle : 0.0, g_kt.io_r, g_kt.io_w, tr::cur_tid(), tr::cur_cpu());
        for (auto& kv : sms) {
            Sm& s = kv.second;
            std::vector<std::pair<long long, int>> edge;
            edge.reserve(2 * s.iv.size());
            double occ_avg[NB] = {0};
            for (const auto& iv : s.iv) {
                edge.push_back({iv.first, 1});
                edge.push_back({iv.second + 1, -1});
                const int b0 = (int)((iv.first - cmin) / bw), b1 = std::min<int>(NB - 1, (int)((iv.second - cmin) / bw));
                for (int b = b0; b <= b1; ++b) {
                    const double lo = cmin + b * bw, hi = lo + bw;
                    occ_avg[b] += (std::min<double>((double)iv.second + 1.0, hi) - std::max<double>((double)iv.first, lo)) / bw;
                }
            }
            std::sort(edge.begin(), edge.end());
            int cur = 0, peak = 0;
            for (const auto& e : edge) { cur += e.second; peak = std::max(peak, cur); }
            char occ[NB * 12 + 8];
            int o = 0;
            for (int b = 0; b < NB; ++b) o += std::snprintf(occ + o, sizeof occ - o, "%s%.1f", b ? "," : "", occ_avg[b]);
            TR_EVENT("gpu.sm", t0 + (uint64_t)((s.c0 - cmin) * ns_per_cycle), (uint64_t)((s.c1 - s.c0) * ns_per_cycle),
                     "\"kernel\":\"%s\",\"sm\":%u,\"blocks\":%zu,\"threads\":%zu,\"busy_ns\":%.0f,\"peak_blocks\":%d,"
                     "\"peak_threads\":%d,\"peak_registers\":%lld,\"avg_blocks_resident\":[%s]",
                     name, kv.first, s.iv.size(), s.iv.size() * (size_t)TPB, s.busy * ns_per_cycle, peak, peak * (int)TPB,
                     (long long)peak * TPB * std::max(a.regs, 0), occ);
        }
        if ((long)rec <= g_kt.detail)
            for (unsigned b = 0; b < rec; ++b) {
                const KtRec& r = g_kt.host[b];
                if (r.c1 < r.c0) continue;
                TR_EVENT("gpu.block", t0 + (uint64_t)((r.c0 - cmin) * ns_per_cycle), (uint64_t)((r.c1 - r.c0) * ns_per_cycle),
                         "\"kernel\":\"%s\",\"block\":%u,\"sm\":%u,\"threads\":%d", name, b, r.sm, (int)TPB);
            }
        g_kt.io_r = g_kt.io_w = 0;
    }
};

#define TR_GPU_IO(r, w) tr_gpu_io((size_t)(r), (size_t)(w))
#define LAUNCH(n, kernel, ...)                                                    \
    do {                                                                          \
        const long long ln_ = (long long)(n);                                     \
        if (ln_ > 0) {                                                            \
            KtLaunch kl_(ln_, kt_attrs(#kernel, kernel));                         \
            RAW_LAUNCH(ln_, kernel, __VA_ARGS__, g_kt.buf, g_kt.cap);             \
            kl_.done();                                                           \
        }                                                                         \
    } while (0)
#else
#define TR_GPU_IO(r, w) ((void)0)
#define LAUNCH(n, kernel, ...)                                 \
    do {                                                       \
        const long long ln_ = (long long)(n);                  \
        if (ln_ > 0) RAW_LAUNCH(ln_, kernel, __VA_ARGS__);     \
    } while (0)
#endif

// ============================================================================
// Device memory helpers
// ============================================================================
static void* dalloc(size_t bytes) {
    void* p = nullptr;
    const double t0 = now_s();
    if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) {
        cudaGetLastError();
        fail("device allocation of %.2f GiB failed", bytes / 1073741824.0);
    }
    (void)t0;
    TR_EVENT("gpu.alloc", tr::now_ns(), (uint64_t)((now_s() - t0) * 1e9), "\"bytes\":%zu,\"addr\":\"%p\",\"tid\":%d,\"cpu\":%d", bytes, p, tr::cur_tid(), tr::cur_cpu());
    return p;
}
static void dfree(void* p) {
    if (!p) return;
    TR_EVENT("gpu.free", tr::now_ns(), 0, "\"addr\":\"%p\"", p);
    cudaFree(p);
}
static void h2d(void* dst, const void* src, size_t bytes) {
    TR_SCOPE("gpu.xfer", "\"dir\":\"host_to_device\",\"bytes\":%zu,\"host\":\"%p\",\"device\":\"%p\",\"tid\":%d,\"cpu\":%d", bytes, src, dst, tr::cur_tid(), tr::cur_cpu());
    CU(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
static void d2h(void* dst, const void* src, size_t bytes) {
    TR_SCOPE("gpu.xfer", "\"dir\":\"device_to_host\",\"bytes\":%zu,\"host\":\"%p\",\"device\":\"%p\",\"tid\":%d,\"cpu\":%d", bytes, dst, src, tr::cur_tid(), tr::cur_cpu());
    CU(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
static void dzero(void* p, size_t bytes) { CU(cudaMemset(p, 0, bytes)); }

// ============================================================================
// Storage formats as seen by device code
// ============================================================================
// st   element type in memory        f(p, i)  raw value as float
// i(p, i)  raw value as int          a(p, i)  raw value in the reduction type
struct PF32 {
    typedef float st; typedef float acc; typedef float wt;
    enum { is_int = 0, packed = 0, bins = 256 };
    static __device__ __forceinline__ float f(const st* p, long long i) { return p[i]; }
    static __device__ __forceinline__ int i(const st* p, long long i) { return (int)p[i]; }
    static __device__ __forceinline__ acc a(const st* p, long long i) { return p[i]; }
    static __device__ __forceinline__ int bin(const st* p, long long i) {
        const int b = (int)((p[i] + 1.0f) * 128.0f);
        return b < 0 ? 0 : b > 255 ? 255 : b;
    }
};
struct PF16 {
    typedef __half st; typedef float acc; typedef float wt;
    enum { is_int = 0, packed = 0, bins = 256 };
    static __device__ __forceinline__ float f(const st* p, long long i) { return __half2float(p[i]); }
    static __device__ __forceinline__ int i(const st* p, long long i) { return (int)__half2float(p[i]); }
    static __device__ __forceinline__ acc a(const st* p, long long i) { return __half2float(p[i]); }
    static __device__ __forceinline__ int bin(const st* p, long long i) {
        const int b = (int)((__half2float(p[i]) + 1.0f) * 128.0f);
        return b < 0 ? 0 : b > 255 ? 255 : b;
    }
};
struct PI8 {
    typedef signed char st; typedef long long acc; typedef signed char wt;
    enum { is_int = 1, packed = 0, bins = 256 };
    static __device__ __forceinline__ float f(const st* p, long long i) { return (float)p[i]; }
    static __device__ __forceinline__ int i(const st* p, long long i) { return p[i]; }
    static __device__ __forceinline__ acc a(const st* p, long long i) { return p[i]; }
    static __device__ __forceinline__ int bin(const st* p, long long i) { return (int)p[i] + 128; }
};
struct PI4 {  // two signed 4-bit values per byte, element 2k in the low nibble
    typedef unsigned char st; typedef long long acc; typedef signed char wt;
    enum { is_int = 1, packed = 1, bins = 16 };
    static __device__ __forceinline__ int i(const st* p, long long i) {
        const int u = (p[i >> 1] >> ((int)(i & 1) * 4)) & 15;
        return (u ^ 8) - 8;
    }
    static __device__ __forceinline__ float f(const st* p, long long j) { return (float)i(p, j); }
    static __device__ __forceinline__ acc a(const st* p, long long j) { return i(p, j); }
    static __device__ __forceinline__ int bin(const st* p, long long j) { return i(p, j) + 8; }
};
struct PI32 {
    typedef int st; typedef long long acc;
    static __device__ __forceinline__ acc a(const st* p, long long i) { return p[i]; }
};
struct PD64 {
    typedef double st; typedef double acc;
    static __device__ __forceinline__ acc a(const st* p, long long i) { return p[i]; }
};

// ============================================================================
// Kernels
// ============================================================================

// Thread t of T sums elements t, t+T, t+2T, ... into out[t].
// mode 0: sum, 1: sum of |x|, 2: sum of x*x.
template <class P> __global__ void k_reduce(long long T, long long n, const typename P::st* x, int mode, double* out KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    typename P::acc a = 0;
    for (long long j = t; j < n; j += T) {
        const typename P::acc v = P::a(x, j);
        a += mode == 0 ? v : mode == 1 ? (v < 0 ? -v : v) : v * v;
    }
    out[t] = (double)a;
}

// Thread t counts elements t, t+T, ... into a private histogram, then merges it.
template <class P> __global__ void k_hist(long long T, long long n, const typename P::st* x, unsigned long long* bins KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    unsigned h[P::bins];
    for (int b = 0; b < P::bins; ++b) h[b] = 0;
    for (long long j = t; j < n; j += T) ++h[P::bin(x, j)];
    for (int b = 0; b < P::bins; ++b)
        if (h[b]) atomicAdd(&bins[b], (unsigned long long)h[b]);
}

// One Jacobi step. Element e of the batch of N x N grids; edges are clamped.
template <class P> static __device__ __forceinline__ float st_f(const typename P::st* in, long long e, int N) {
    const int x = (int)(e % N), y = (int)((e / N) % N);
    const float c = P::f(in, e);
    const float n = P::f(in, y > 0 ? e - N : e), s = P::f(in, y < N - 1 ? e + N : e);
    const float w = P::f(in, x > 0 ? e - 1 : e), ea = P::f(in, x < N - 1 ? e + 1 : e);
    return 0.5f * c + 0.125f * (n + s + w + ea);
}
template <class P> static __device__ __forceinline__ int st_i(const typename P::st* in, long long e, int N) {
    const int x = (int)(e % N), y = (int)((e / N) % N);
    const int c = P::i(in, e);
    const int n = P::i(in, y > 0 ? e - N : e), s = P::i(in, y < N - 1 ? e + N : e);
    const int w = P::i(in, x > 0 ? e - 1 : e), ea = P::i(in, x < N - 1 ? e + 1 : e);
    return (4 * c + n + s + w + ea) >> 3;
}
static __device__ __forceinline__ void st_put(float* out, long long t, float v, int, int) { out[t] = v; }
static __device__ __forceinline__ void st_put(__half* out, long long t, float v, int, int) { out[t] = __float2half(v); }
static __device__ __forceinline__ void st_put(signed char* out, long long t, float, int a, int) { out[t] = (signed char)a; }
static __device__ __forceinline__ void st_put(unsigned char* out, long long t, float, int a, int b) { out[t] = (unsigned char)((a & 15) | ((b & 15) << 4)); }

// Thread t writes one storage unit: one cell, or the two cells of an int4 byte.
template <class P> __global__ void k_stencil(long long units, const typename P::st* in, typename P::st* out, int N KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= units) return;
    if (P::packed) st_put(out, t, 0.0f, st_i<P>(in, 2 * t, N), st_i<P>(in, 2 * t + 1, N));
    else if (P::is_int) st_put(out, t, 0.0f, st_i<P>(in, t, N), 0);
    else st_put(out, t, st_f<P>(in, t, N), 0, 0);
}

// Softmax over rows of V values, in five data-parallel steps.
template <class P> __global__ void k_sm_chunkmax(long long T, const typename P::st* x, int V, int nch, float gain, float* tmp KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const long long r = t / nch;
    const int j0 = (int)(t % nch) * SMCH, j1 = j0 + SMCH < V ? j0 + SMCH : V;
    float m = -3.0e38f;
    for (int j = j0; j < j1; ++j) m = fmaxf(m, P::f(x, r * V + j) * gain);
    tmp[t] = m;
}
__global__ void k_sm_rowred(long long R, const float* tmp, int nch, int is_sum, float* out KT_PARAM) {
    KT_SCOPE;
    const long long r = gid();
    if (r >= R) return;
    float a = is_sum ? 0.0f : -3.0e38f;
    for (int c = 0; c < nch; ++c) a = is_sum ? a + tmp[r * nch + c] : fmaxf(a, tmp[r * nch + c]);
    out[r] = a;
}
template <class P> __global__ void k_sm_exp(long long T, const typename P::st* x, int V, float gain, const float* rmax, float* out KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    out[t] = expf(P::f(x, t) * gain - rmax[t / V]);
}
__global__ void k_sm_chunksum(long long T, const float* e, int V, int nch, float* tmp KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const long long r = t / nch;
    const int j0 = (int)(t % nch) * SMCH, j1 = j0 + SMCH < V ? j0 + SMCH : V;
    float s = 0.0f;
    for (int j = j0; j < j1; ++j) s += e[r * V + j];
    tmp[t] = s;
}
__global__ void k_sm_scale(long long T, float* e, int V, const float* rsum KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    e[t] /= rsum[t / V];
}

// Format conversions.
template <class P> __global__ void k_tofloat(long long n, const typename P::st* s, float* d KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= n) return;
    d[t] = P::f(s, t);
}
__global__ void k_unpack4(long long nbytes, const unsigned char* s, signed char* d KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= nbytes) return;
    d[2 * t] = (signed char)(((s[t] & 15) ^ 8) - 8);
    d[2 * t + 1] = (signed char)(((s[t] >> 4) ^ 8) - 8);
}
__global__ void k_fill(long long n, float* p, float v KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= n) return;
    p[t] = v;
}

// Matrix product without cuBLAS: thread t computes Y[token][m] = X[token] . W[m].
// Used for formats or systems where the cuBLAS routine is not available.
template <class P> __global__ void k_lin(long long T, const typename P::st* X, const typename P::st* W, void* Y, int K, int M, int acc KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const long long xo = (t / M) * K, wo = (t % M) * K;
    if (P::is_int) {
        int s = 0;
        for (int k = 0; k < K; ++k) s += P::i(X, xo + k) * P::i(W, wo + k);
        int* y = (int*)Y + t;
        *y = acc ? *y + s : s;
    } else {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += P::f(X, xo + k) * P::f(W, wo + k);
        float* y = (float*)Y + t;
        *y = acc ? *y + s : s;
    }
}

// Direct 3x3 convolution, zero padding: thread t computes one output value.
template <class P> __global__ void k_conv(long long T, const typename P::st* X, const typename P::wt* Wt, void* Out, int C, int H, int Wd KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const int x = (int)(t % Wd), y = (int)((t / Wd) % H), co = (int)((t / ((long long)Wd * H)) % C);
    const long long b = t / ((long long)Wd * H * C);
    float sf = 0.0f;
    int si = 0;
    for (int ci = 0; ci < C; ++ci) {
        const typename P::wt* w = Wt + ((long long)co * C + ci) * 9;
        const long long plane = (b * C + ci) * (long long)H;
        for (int ky = 0; ky < 3; ++ky) {
            const int yy = y + ky - 1;
            if (yy < 0 || yy >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                const int xx = x + kx - 1;
                if (xx < 0 || xx >= Wd) continue;
                const long long idx = (plane + yy) * Wd + xx;
                if (P::is_int) si += (int)w[ky * 3 + kx] * P::i(X, idx);
                else sf += (float)w[ky * 3 + kx] * P::f(X, idx);
            }
        }
    }
    if (P::is_int) ((int*)Out)[t] = si;
    else ((float*)Out)[t] = sf;
}

// FlashAttention. Thread t owns query row t (sequence t / L). Scores for a tile
// of FBC keys are held in thread-local memory; the L x L matrix never exists.
template <class P> static __device__ __forceinline__ float fl_score(const typename P::st* Q, const typename P::st* K, long long qi, long long kj, int d, float alpha) {
    if (P::is_int) {
        int s = 0;
        for (int c = 0; c < d; ++c) s += P::i(Q, qi + c) * P::i(K, kj + c);
        return (float)s * alpha;
    }
    float s = 0.0f;
    for (int c = 0; c < d; ++c) s += P::f(Q, qi + c) * P::f(K, kj + c);
    return s * alpha;
}

// v2: the whole row in one launch; m, l and the unnormalised output stay in the
// thread, and the output is divided by l once at the end.
template <class P> __global__ void k_fl2(long long T, const typename P::st* Q, const typename P::st* K, const typename P::st* V, float* O, int L, int d, float alpha, float vscale KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const long long base = (t / L) * (long long)L * d, qi = t * (long long)d;
    float o[FLASH_MAXD], s[FBC];
    for (int c = 0; c < d; ++c) o[c] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (int j0 = 0; j0 < L; j0 += FBC) {
        const int bc = j0 + FBC < L ? FBC : L - j0;
        float mt = -3.0e38f;
        for (int j = 0; j < bc; ++j) { s[j] = fl_score<P>(Q, K, qi, base + (long long)(j0 + j) * d, d, alpha); mt = fmaxf(mt, s[j]); }
        if (mt > m) {
            const float a = expf(m - mt);
            l *= a;
            for (int c = 0; c < d; ++c) o[c] *= a;
            m = mt;
        }
        for (int j = 0; j < bc; ++j) {
            const float p = expf(s[j] - m);
            l += p;
            const long long vj = base + (long long)(j0 + j) * d;
            for (int c = 0; c < d; ++c) o[c] += p * P::f(V, vj + c);
        }
    }
    const float inv = vscale / l;
    for (int c = 0; c < d; ++c) O[qi + c] = o[c] * inv;
}

// v1: one launch per key/value tile. m, l and the normalised output live in
// device memory and are read, renormalised and written back at every step.
template <class P> __global__ void k_fl1(long long T, const typename P::st* Q, const typename P::st* K, const typename P::st* V, float* O, float* Mx, float* Ls, int L, int d, int j0, int bc, float alpha, float vscale KT_PARAM) {
    KT_SCOPE;
    const long long t = gid();
    if (t >= T) return;
    const long long base = (t / L) * (long long)L * d, qi = t * (long long)d;
    float s[FBC];
    float mt = -3.0e38f;
    for (int j = 0; j < bc; ++j) { s[j] = fl_score<P>(Q, K, qi, base + (long long)(j0 + j) * d, d, alpha); mt = fmaxf(mt, s[j]); }
    const float m_old = Mx[t], l_old = Ls[t];
    const float m_new = fmaxf(m_old, mt), a = expf(m_old - m_new);
    float psum = 0.0f;
    for (int j = 0; j < bc; ++j) { s[j] = expf(s[j] - m_new); psum += s[j]; }
    const float l_new = a * l_old + psum, f = l_old * a, inv = 1.0f / l_new;
    for (int c = 0; c < d; ++c) {
        float acc = 0.0f;
        for (int j = 0; j < bc; ++j) acc += s[j] * P::f(V, base + (long long)(j0 + j) * d + c);
        O[qi + c] = (f * O[qi + c] + vscale * acc) * inv;
    }
    Mx[t] = m_new;
    Ls[t] = l_new;
}

// ============================================================================
// Host-side building blocks
// ============================================================================

template <class P> static double dev_reduce(const typename P::st* x, long long n, int mode) {
    if (n <= 0) return 0.0;
    const long long T1 = std::min<long long>(n, RED_T1), T2 = std::min<long long>(T1, RED_T2);
    LAUNCH(T1, k_reduce<P>, T1, n, x, mode, G.red);
    LAUNCH(T2, k_reduce<PD64>, T2, T1, G.red, 0, G.red + RED_T1);
    double h[RED_T2], s = 0.0;
    d2h(h, G.red + RED_T1, (size_t)T2 * sizeof(double));
    for (long long i = 0; i < T2; ++i) s += h[i];
    return s;
}

// Softmax of R rows of V values read from x (scaled by gain) into e; e may be x.
// tmp needs R * ceil(V / SMCH) floats, rmax and rsum R floats each.
template <class P> static void dev_softmax(const typename P::st* x, long long R, int V, float gain, float* e, float* tmp, float* rmax, float* rsum) {
    const int nch = (V + SMCH - 1) / SMCH;
    LAUNCH(R * nch, k_sm_chunkmax<P>, R * nch, x, V, nch, gain, tmp);
    LAUNCH(R, k_sm_rowred, R, tmp, nch, 0, rmax);
    LAUNCH(R * V, k_sm_exp<P>, R * V, x, V, gain, rmax, e);
    LAUNCH(R * nch, k_sm_chunksum, R * nch, e, V, nch, tmp);
    LAUNCH(R, k_sm_rowred, R, tmp, nch, 1, rsum);
    LAUNCH(R * V, k_sm_scale, R * V, e, V, rsum);
}

#define BLAS_OK(st, what) do { if ((st) != CUBLAS_STATUS_SUCCESS) fail("cuBLAS %s failed (status %d)", what, (int)(st)); } while (0)

// C = alpha * A^T * B + beta * C in float, column-major.
static void dev_sgemm_t(int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    TR_SCOPE("gpu.blas", "\"routine\":\"cublasSgemm\",\"m\":%d,\"n\":%d,\"k\":%d,\"tid\":%d,\"cpu\":%d", m, n, k, tr::cur_tid(), tr::cur_cpu());
    BLAS_OK(cublasSgemm(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc), "sgemm");
    if (TR_ON) gsync();
}
// `count` independent products, operands `stride` elements apart.
static void dev_sgemm_batched(cublasOperation_t ta, int m, int n, int k, float alpha, const float* A, int lda, long long sa,
                              const float* B, int ldb, long long sb, float* C, int ldc, long long sc, int count) {
    TR_SCOPE("gpu.blas", "\"routine\":\"cublasSgemmStridedBatched\",\"m\":%d,\"n\":%d,\"k\":%d,\"batch\":%d,\"tid\":%d,\"cpu\":%d", m, n, k, count, tr::cur_tid(), tr::cur_cpu());
    const float zero = 0.0f;
    if (G.use_sb) BLAS_OK(cublasSgemmStridedBatched(G.blas, ta, CUBLAS_OP_N, m, n, k, &alpha, A, lda, sa, B, ldb, sb, &zero, C, ldc, sc, count), "sgemmStridedBatched");
    else
        for (int i = 0; i < count; ++i)
            BLAS_OK(cublasSgemm(G.blas, ta, CUBLAS_OP_N, m, n, k, &alpha, A + i * sa, lda, B + i * sb, ldb, &zero, C + i * sc, ldc), "sgemm");
    if (TR_ON) gsync();
}
static cublasStatus_t dev_hgemm_t(int m, int n, int k, const __half* A, int lda, const __half* B, int ldb, bool acc, __half* C, int ldc) {
    TR_SCOPE("gpu.blas", "\"routine\":\"cublasHgemm\",\"m\":%d,\"n\":%d,\"k\":%d,\"tid\":%d,\"cpu\":%d", m, n, k, tr::cur_tid(), tr::cur_cpu());
    const __half one = __float2half(1.0f), beta = __float2half(acc ? 1.0f : 0.0f);
    const cublasStatus_t st = cublasHgemm(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, lda, B, ldb, &beta, C, ldc);
    if (TR_ON) gsync();
    return st;
}
static cublasStatus_t dev_i8gemm_t(int m, int n, int k, const signed char* A, int lda, const signed char* B, int ldb, bool acc, int* C, int ldc) {
    TR_SCOPE("gpu.blas", "\"routine\":\"cublasGemmEx(int8->int32)\",\"m\":%d,\"n\":%d,\"k\":%d,\"tid\":%d,\"cpu\":%d", m, n, k, tr::cur_tid(), tr::cur_cpu());
    const int one = 1, beta = acc ? 1 : 0;
    const cublasStatus_t st = cublasGemmEx(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, CUDA_R_8I, lda, B, CUDA_R_8I, ldb, &beta, C,
                                           CUDA_R_32I, ldc, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (TR_ON) gsync();
    return st;
}

// ============================================================================
// Backends
// ============================================================================

struct GpuBase : Backend {
    Case c;
    long long U = 0;
    explicit GpuBase(const Case& cc) : c(cc) {}
    const char* impl() const override { return "cuda"; }
    size_t op_bytes(int op, long long units) const {
        const Operand& o = c.op[op];
        return store_bytes(o.store, o.shared ? o.shared : o.per_unit * (size_t)units);
    }
    void* take(const Staging& s, int op, long long u0, long long u1) const {
        const Operand& o = c.op[op];
        const size_t off = o.shared ? 0 : store_bytes(o.store, o.per_unit * (size_t)u0);
        const size_t bytes = op_bytes(op, u1 - u0);
        void* p = dalloc(bytes);
        h2d(p, (const char*)s.p[op] + off, bytes);
        return p;
    }
};

// ---- 1. sum ----------------------------------------------------------------------
template <class P> struct GpuSum : GpuBase {
    typename P::st* x = nullptr;
    double result = 0.0;
    using GpuBase::GpuBase;
    size_t bytes(long long units) const override { return op_bytes(0, units); }
    void load(const Staging& s, long long u0, long long u1) override { U = u1 - u0; x = (typename P::st*)take(s, 0, u0, u1); gsync(); }
    void unload() override { dfree(x); x = nullptr; }
    double checksum() override { return result; }
    void pass() override {
        TR_GPU_IO(op_bytes(0, U), 0);
        result = dev_reduce<P>(x, U * (long long)SUM_UNIT, 0);
    }
};

// ---- 2. histogram ------------------------------------------------------------------
template <class P> struct GpuHist : GpuBase {
    typename P::st* x = nullptr;
    unsigned long long* bins = nullptr;
    unsigned long long host[256];
    using GpuBase::GpuBase;
    size_t bytes(long long units) const override { return op_bytes(0, units) + 2048; }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        x = (typename P::st*)take(s, 0, u0, u1);
        bins = (unsigned long long*)dalloc(256 * sizeof(unsigned long long));
        gsync();
    }
    void unload() override { dfree(x); dfree(bins); x = nullptr; bins = nullptr; }
    double checksum() override {
        double s = 0.0;
        for (int i = 0; i < 256; ++i) s += (double)host[i] * (i + 1);
        return s;
    }
    void pass() override {
        const long long n = U * (long long)SUM_UNIT, T = std::min<long long>(n, HIST_T);
        dzero(bins, 256 * sizeof(unsigned long long));
        TR_GPU_IO(op_bytes(0, U), 256 * 8);
        LAUNCH(T, k_hist<P>, T, n, x, bins);
        d2h(host, bins, sizeof host);
    }
};

// ---- 3. stencil --------------------------------------------------------------------
template <class P> struct GpuStencil : GpuBase {
    typename P::st *cur = nullptr, *nxt = nullptr;
    using GpuBase::GpuBase;
    size_t bytes(long long units) const override { return 2 * op_bytes(0, units); }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        cur = (typename P::st*)take(s, 0, u0, u1);
        nxt = (typename P::st*)dalloc(op_bytes(0, U));
        gsync();
    }
    void unload() override { dfree(cur); dfree(nxt); cur = nxt = nullptr; }
    double checksum() override { return dev_reduce<P>(cur, U * c.s.b * c.s.b, 1); }
    void pass() override {
        const int N = (int)c.s.b;
        const long long cells = U * (long long)N * N, units = P::packed ? cells / 2 : cells;
        for (int step = 0; step < (int)c.s.c; ++step) {
            TR_GPU_IO(op_bytes(0, U), op_bytes(0, U));
            LAUNCH(units, k_stencil<P>, units, cur, nxt, N);
            std::swap(cur, nxt);
        }
        gsync();
    }
};

// ---- 4. softmax ----------------------------------------------------------------------
template <class P> struct GpuSoftmax : GpuBase {
    typename P::st* x = nullptr;
    float *out = nullptr, *tmp = nullptr, *rmax = nullptr, *rsum = nullptr;
    using GpuBase::GpuBase;
    size_t nch() const { return ((size_t)c.s.b + SMCH - 1) / SMCH; }
    size_t bytes(long long units) const override { return op_bytes(0, units) + (size_t)units * (c.s.b + nch() + 2) * 4; }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        x = (typename P::st*)take(s, 0, u0, u1);
        out = (float*)dalloc((size_t)U * c.s.b * 4);
        tmp = (float*)dalloc((size_t)U * nch() * 4);
        rmax = (float*)dalloc((size_t)U * 4);
        rsum = (float*)dalloc((size_t)U * 4);
        gsync();
    }
    void unload() override { dfree(x); dfree(out); dfree(tmp); dfree(rmax); dfree(rsum); x = nullptr; out = tmp = rmax = rsum = nullptr; }
    double checksum() override { return dev_reduce<PF32>(out, U * c.s.b, 2); }
    void pass() override {
        TR_GPU_IO(op_bytes(0, U), (size_t)U * c.s.b * 4);
        dev_softmax<P>(x, U, (int)c.s.b, LOGIT_GAIN * int_scale(c.prec), out, tmp, rmax, rsum);
        gsync();
    }
};

// ---- 5. linear -----------------------------------------------------------------------
// fp32 -> cublasSgemm. fp16 -> cublasHgemm, result kept in half. int8 ->
// cublasGemmEx with int32 accumulation. int4 -> unpack kernel, then the int8
// path. A format whose cuBLAS routine failed its start-up probe uses k_lin.
struct GpuLinear : GpuBase {
    void *X = nullptr, *W = nullptr, *Y = nullptr;
    signed char* W8 = nullptr;
    bool y_half = false;  // fp16 result held in half (cuBLAS) or float (kernel); fixed when the case is loaded
    using GpuBase::GpuBase;
    bool lib() const { return c.prec == F32 || (c.prec == F16 && G.use_h) || (c.prec >= I8 && G.use_i8); }
    const char* impl() const override { return lib() ? "cublas" : "cuda"; }
    size_t bytes(long long units) const override {
        return op_bytes(0, units) + op_bytes(1, units) + (size_t)units * c.s.c * 4 + (c.prec == I4 ? (size_t)c.s.c * c.s.b : 0);
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        X = take(s, 0, u0, u1);
        W = take(s, 1, u0, u1);
        Y = dalloc((size_t)U * c.s.c * 4);
        if (c.prec == I4) W8 = (signed char*)dalloc((size_t)c.s.c * c.s.b);
        y_half = c.prec == F16 && G.use_h;
        gsync();
    }
    void unload() override { dfree(X); dfree(W); dfree(Y); dfree(W8); X = W = Y = nullptr; W8 = nullptr; }
    double checksum() override {
        const long long n = U * c.s.c;
        if (c.prec >= I8) return dev_reduce<PI32>((const int*)Y, n, 1);
        if (y_half) return dev_reduce<PF16>((const __half*)Y, n, 1);
        return dev_reduce<PF32>((const float*)Y, n, 1);
    }
    void pass() override {
        const int K = (int)c.s.b, M = (int)c.s.c, T = (int)U;
        const size_t wb = store_bytes(c.prec, (size_t)M * K);
        for (int l = 0; l < (int)c.s.d; ++l) {
            const char* w = (const char*)W + (size_t)l * wb;
            const bool acc = l > 0;
            TR_GPU_IO(wb + op_bytes(0, U), (size_t)U * M * 4);
            if (c.prec == F32) dev_sgemm_t(M, T, K, 1.0f, (const float*)w, K, (const float*)X, K, acc ? 1.0f : 0.0f, (float*)Y, M);
            else if (c.prec == F16) {
                if (y_half) {
                    const cublasStatus_t st = dev_hgemm_t(M, T, K, (const __half*)w, K, (const __half*)X, K, acc, (__half*)Y, M);
                    if (st != CUBLAS_STATUS_SUCCESS) { G.use_h = false; BLAS_OK(st, "hgemm"); }  // later cases use the kernel
                } else LAUNCH((long long)T * M, k_lin<PF16>, (long long)T * M, (const __half*)X, (const __half*)w, Y, K, M, (int)acc);
            } else {
                const signed char* w8 = (const signed char*)w;
                if (c.prec == I4) {
                    LAUNCH((long long)M * K / 2, k_unpack4, (long long)M * K / 2, (const unsigned char*)w, W8);
                    w8 = W8;
                }
                // Both paths write the same int32 result, so a refusal by cuBLAS for this
                // shape is answered by the kernel for this one product.
                if (!G.use_i8 || dev_i8gemm_t(M, T, K, w8, K, (const signed char*)X, K, acc, (int*)Y, M) != CUBLAS_STATUS_SUCCESS)
                    LAUNCH((long long)T * M, k_lin<PI8>, (long long)T * M, (const signed char*)X, w8, Y, K, M, (int)acc);
            }
        }
        gsync();
    }
};

// ---- 6. conv2d -----------------------------------------------------------------------
struct GpuConv : GpuBase {
    void *X = nullptr, *Wraw = nullptr, *Wc = nullptr, *Out = nullptr;
    using GpuBase::GpuBase;
    size_t nw() const { return (size_t)c.s.b * c.s.b * 9; }
    size_t nout(long long units) const { return (size_t)units * c.s.b * c.s.c * c.s.c; }
    size_t bytes(long long units) const override { return op_bytes(0, units) + nout(units) * 4 + nw() * 5; }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        X = take(s, 0, u0, u1);
        Wraw = take(s, 1, u0, u1);
        Out = dalloc(nout(U) * 4);
        // Kernel weights are widened once: half -> float, int4 -> int8.
        if (c.prec == F16) { Wc = dalloc(nw() * 4); LAUNCH(nw(), k_tofloat<PF16>, (long long)nw(), (const __half*)Wraw, (float*)Wc); }
        else if (c.prec == I4) { Wc = dalloc(nw()); LAUNCH(nw() / 2, k_unpack4, (long long)(nw() / 2), (const unsigned char*)Wraw, (signed char*)Wc); }
        gsync();
    }
    void unload() override { dfree(X); dfree(Wraw); dfree(Wc); dfree(Out); X = Wraw = Wc = Out = nullptr; }
    double checksum() override {
        return c.prec >= I8 ? dev_reduce<PI32>((const int*)Out, (long long)nout(U), 1) : dev_reduce<PF32>((const float*)Out, (long long)nout(U), 1);
    }
    void pass() override {
        const int C = (int)c.s.b, HW = (int)c.s.c;
        const long long T = (long long)nout(U);
        TR_GPU_IO(op_bytes(0, U) + nw() * 4, nout(U) * 4);
        if (c.prec == F32) LAUNCH(T, k_conv<PF32>, T, (const float*)X, (const float*)Wraw, Out, C, HW, HW);
        else if (c.prec == F16) LAUNCH(T, k_conv<PF16>, T, (const __half*)X, (const float*)Wc, Out, C, HW, HW);
        else LAUNCH(T, k_conv<PI8>, T, (const signed char*)X, (const signed char*)(c.prec == I4 ? Wc : Wraw), Out, C, HW, HW);
        gsync();
    }
};

// ---- 7. attention, scores materialised --------------------------------------------------
// Q, K, V widened to float; scores and outputs are batched cuBLAS products; as
// many sequences at a time as fit in the score scratch.
template <class P> struct GpuAttn : GpuBase {
    typename P::st *Q = nullptr, *K = nullptr, *V = nullptr;
    float *Qf = nullptr, *Kf = nullptr, *Vf = nullptr, *O = nullptr, *S = nullptr, *tmp = nullptr, *rmax = nullptr, *rsum = nullptr;
    using GpuBase::GpuBase;
    const char* impl() const override { return "cublas"; }
    long long chunk(long long units) const {
        const double per = 4.0 * (double)c.s.b * (double)c.s.b;
        return std::max<long long>(1, std::min<long long>(units, (long long)(8.0 * 1073741824.0 / per)));
    }
    size_t bytes(long long units) const override {
        const size_t L = c.s.b, d = c.s.c, uc = (size_t)chunk(units);
        return 3 * op_bytes(0, units) + (size_t)units * L * d * 4 * (c.prec != F32 ? 4 : 1) + uc * L * (L + (L + SMCH - 1) / SMCH + 2) * 4;
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        const size_t L = c.s.b, d = c.s.c, n = (size_t)U * L * d, uc = (size_t)chunk(U);
        Q = (typename P::st*)take(s, 0, u0, u1);
        K = (typename P::st*)take(s, 1, u0, u1);
        V = (typename P::st*)take(s, 2, u0, u1);
        O = (float*)dalloc(n * 4);
        if (c.prec != F32) { Qf = (float*)dalloc(n * 4); Kf = (float*)dalloc(n * 4); Vf = (float*)dalloc(n * 4); }
        S = (float*)dalloc(uc * L * L * 4);
        tmp = (float*)dalloc(uc * L * ((L + SMCH - 1) / SMCH) * 4);
        rmax = (float*)dalloc(uc * L * 4);
        rsum = (float*)dalloc(uc * L * 4);
        gsync();
    }
    void unload() override {
        dfree(Q); dfree(K); dfree(V); dfree(Qf); dfree(Kf); dfree(Vf); dfree(O); dfree(S); dfree(tmp); dfree(rmax); dfree(rsum);
        Q = K = V = nullptr; Qf = Kf = Vf = O = S = tmp = rmax = rsum = nullptr;
    }
    double checksum() override { return dev_reduce<PF32>(O, U * c.s.b * c.s.c, 1); }
    void pass() override {
        const int L = (int)c.s.b, d = (int)c.s.c;
        const long long ld = (long long)L * d, n = U * ld, uc = chunk(U);
        const float sI = int_scale(c.prec), alpha = sI * sI / sqrtf((float)d);
        const float *q = (const float*)Q, *k = (const float*)K, *v = (const float*)V;
        if (c.prec != F32) {
            LAUNCH(n, k_tofloat<P>, n, Q, Qf);
            LAUNCH(n, k_tofloat<P>, n, K, Kf);
            LAUNCH(n, k_tofloat<P>, n, V, Vf);
            q = Qf; k = Kf; v = Vf;
        }
        for (long long u0 = 0; u0 < U; u0 += uc) {
            const int nb = (int)std::min<long long>(uc, U - u0);
            TR_GPU_IO((size_t)nb * ld * 12, (size_t)nb * L * L * 4 + (size_t)nb * ld * 4);
            // S[i*L + j] = alpha * q_i . k_j
            dev_sgemm_batched(CUBLAS_OP_T, L, L, d, alpha, k + u0 * ld, d, ld, q + u0 * ld, d, ld, S, L, (long long)L * L, nb);
            dev_softmax<PF32>(S, (long long)nb * L, L, 1.0f, S, tmp, rmax, rsum);
            // o_i = sum_j p_ij v_j
            dev_sgemm_batched(CUBLAS_OP_N, d, L, L, sI, v + u0 * ld, d, ld, S, L, (long long)L * L, O + u0 * ld, d, ld, nb);
        }
        gsync();
    }
};

// ---- 8, 9. FlashAttention v1 and v2 -------------------------------------------------------
template <class P> struct GpuFlash : GpuBase {
    typename P::st *Q = nullptr, *K = nullptr, *V = nullptr;
    float *O = nullptr, *Mx = nullptr, *Ls = nullptr;
    int version;
    GpuFlash(const Case& cc, int v) : GpuBase(cc), version(v) {}
    size_t bytes(long long units) const override {
        return 3 * op_bytes(0, units) + (size_t)units * c.s.b * c.s.c * 4 + (version == 1 ? (size_t)units * c.s.b * 8 : 0);
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        if (c.s.c > FLASH_MAXD) fail("flash attention supports head dimensions up to %d", (int)FLASH_MAXD);
        Q = (typename P::st*)take(s, 0, u0, u1);
        K = (typename P::st*)take(s, 1, u0, u1);
        V = (typename P::st*)take(s, 2, u0, u1);
        O = (float*)dalloc((size_t)U * c.s.b * c.s.c * 4);
        if (version == 1) { Mx = (float*)dalloc((size_t)U * c.s.b * 4); Ls = (float*)dalloc((size_t)U * c.s.b * 4); }
        gsync();
    }
    void unload() override { dfree(Q); dfree(K); dfree(V); dfree(O); dfree(Mx); dfree(Ls); Q = K = V = nullptr; O = Mx = Ls = nullptr; }
    double checksum() override { return dev_reduce<PF32>(O, U * c.s.b * c.s.c, 1); }
    void pass() override {
        const int L = (int)c.s.b, d = (int)c.s.c;
        const long long T = U * L;
        const float sI = int_scale(c.prec), alpha = sI * sI / sqrtf((float)d);
        const size_t in_b = 3 * op_bytes(0, U), out_b = (size_t)T * d * 4;
        (void)in_b;
        if (version == 2) {
            TR_GPU_IO(in_b, out_b);
            LAUNCH(T, k_fl2<P>, T, Q, K, V, O, L, d, alpha, sI);
        } else {
            LAUNCH(T, k_fill, T, Mx, -1e30f);
            dzero(Ls, (size_t)T * 4);
            dzero(O, out_b);
            for (int j0 = 0; j0 < L; j0 += FBC) {
                const int bc = std::min<int>(FBC, L - j0);
                TR_GPU_IO(op_bytes(0, U) + 2 * store_bytes(c.prec, (size_t)U * bc * d) + out_b + (size_t)T * 8, out_b + (size_t)T * 8);
                LAUNCH(T, k_fl1<P>, T, Q, K, V, O, Mx, Ls, L, d, j0, bc, alpha, sI);
            }
        }
        gsync();
    }
};

template <template <class> class B, class... A> static Backend* by_prec(const Case& c, A... a) {
    switch (c.prec) {
        case F32: return new B<PF32>(c, a...);
        case F16: return new B<PF16>(c, a...);
        case I8: return new B<PI8>(c, a...);
        default: return new B<PI4>(c, a...);
    }
}

Backend* make_gpu_backend(const Case& c) {
    if (!G.ok) return nullptr;
    switch (c.task) {
        case T_SUM: return by_prec<GpuSum>(c);
        case T_HIST: return by_prec<GpuHist>(c);
        case T_STENCIL: return by_prec<GpuStencil>(c);
        case T_SOFTMAX: return by_prec<GpuSoftmax>(c);
        case T_LINEAR: return new GpuLinear(c);
        case T_CONV: return new GpuConv(c);
        case T_ATTN: return by_prec<GpuAttn>(c);
        case T_FLASH1: return by_prec<GpuFlash>(c, 1);
        default: return by_prec<GpuFlash>(c, 2);
    }
}

// ============================================================================
// Start-up: device, cuBLAS, and a probe of each cuBLAS routine against a host
// reference, so a routine that is missing or behaves differently on this
// system is replaced by the plain kernel instead of producing wrong numbers.
// ============================================================================
static bool probe_blas() {
    enum { M = 8, T = 4, K = 12, NB = 3 };  // multiples of 4, which the int8 routine may require
    int a[M * K], b[T * K], ref[T * M];
    for (int i = 0; i < M * K; ++i) a[i] = (i * 7 + 3) % 15 - 7;
    for (int i = 0; i < T * K; ++i) b[i] = (i * 5 + 1) % 13 - 6;
    for (int t = 0; t < T; ++t)
        for (int m = 0; m < M; ++m) {
            int s = 0;
            for (int k = 0; k < K; ++k) s += b[t * K + k] * a[m * K + k];
            ref[t * M + m] = s;
        }
    void *dA = dalloc(M * K * 4 * NB), *dB = dalloc(T * K * 4 * NB), *dC = dalloc(T * M * 4 * NB);
    bool all = true;
    try {
        {   // half
            __half ha[M * K], hb[T * K], hc[T * M];
            for (int i = 0; i < M * K; ++i) ha[i] = __float2half(a[i] / 8.0f);
            for (int i = 0; i < T * K; ++i) hb[i] = __float2half(b[i] / 8.0f);
            h2d(dA, ha, sizeof ha); h2d(dB, hb, sizeof hb);
            // Twice: C = A^T B, then C += A^T B, the two forms the linear workload uses.
            bool ok = dev_hgemm_t(M, T, K, (const __half*)dA, K, (const __half*)dB, K, false, (__half*)dC, M) == CUBLAS_STATUS_SUCCESS &&
                      dev_hgemm_t(M, T, K, (const __half*)dA, K, (const __half*)dB, K, true, (__half*)dC, M) == CUBLAS_STATUS_SUCCESS;
            if (ok) {
                d2h(hc, dC, sizeof hc);
                for (int i = 0; i < T * M; ++i) ok = ok && fabsf(__half2float(hc[i]) - 2.0f * ref[i] / 64.0f) < 2e-2f;
            }
            G.use_h = ok;
        }
        {   // int8 -> int32
            signed char ia[M * K], ib[T * K];
            int ic[T * M];
            for (int i = 0; i < M * K; ++i) ia[i] = (signed char)a[i];
            for (int i = 0; i < T * K; ++i) ib[i] = (signed char)b[i];
            h2d(dA, ia, sizeof ia); h2d(dB, ib, sizeof ib);
            bool ok = dev_i8gemm_t(M, T, K, (const signed char*)dA, K, (const signed char*)dB, K, false, (int*)dC, M) == CUBLAS_STATUS_SUCCESS &&
                      dev_i8gemm_t(M, T, K, (const signed char*)dA, K, (const signed char*)dB, K, true, (int*)dC, M) == CUBLAS_STATUS_SUCCESS;
            if (ok) {
                d2h(ic, dC, sizeof ic);
                for (int i = 0; i < T * M; ++i) ok = ok && ic[i] == 2 * ref[i];
            }
            G.use_i8 = ok;
        }
        {   // strided batch, float
            float fa[M * K * NB], fb[T * K * NB], fc[T * M * NB];
            for (int n = 0; n < NB; ++n) {
                for (int i = 0; i < M * K; ++i) fa[n * M * K + i] = (float)(a[i] + n);
                for (int i = 0; i < T * K; ++i) fb[n * T * K + i] = (float)b[i];
            }
            h2d(dA, fa, sizeof fa); h2d(dB, fb, sizeof fb);
            const float one = 1.0f, zero = 0.0f;
            bool ok = cublasSgemmStridedBatched(G.blas, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K, &one, (const float*)dA, K, M * K, (const float*)dB, K,
                                                T * K, &zero, (float*)dC, M, T * M, NB) == CUBLAS_STATUS_SUCCESS;
            if (ok) {
                d2h(fc, dC, sizeof fc);
                for (int n = 0; n < NB; ++n)
                    for (int t = 0; t < T; ++t)
                        for (int m = 0; m < M; ++m) {
                            float s = 0.0f;
                            for (int k = 0; k < K; ++k) s += fb[n * T * K + t * K + k] * fa[n * M * K + m * K + k];
                            ok = ok && fabsf(fc[n * T * M + t * M + m] - s) < 1e-3f;
                        }
            }
            G.use_sb = ok;
        }
    } catch (const std::exception&) {
        all = false;
    }
    cudaGetLastError();
    dfree(dA); dfree(dB); dfree(dC);
    return all;
}

bool gpu_startup(std::string& describe, std::string& why) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        why = "no CUDA device";
        return false;
    }
    try {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);  // waiting on the GPU must not occupy a CPU core
        cudaGetLastError();
        CU(cudaSetDevice(0));
        cudaDeviceProp prop;
        CU(cudaGetDeviceProperties(&prop, 0));
        G.name = prop.name;
        if (cudaDeviceGetAttribute(&G.sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) { cudaGetLastError(); G.sms = 0; }
        if (cublasCreate(&G.blas) != CUBLAS_STATUS_SUCCESS) fail("cublasCreate failed");
        G.red = (double*)dalloc((RED_T1 + RED_T2) * sizeof(double));
#ifdef GB10_TRACE
        g_kt.buf = (KtRec*)dalloc((size_t)g_kt.cap * sizeof(KtRec));
        if (cudaDeviceGetAttribute(&g_kt.clock_khz, cudaDevAttrClockRate, 0) != cudaSuccess) { cudaGetLastError(); g_kt.clock_khz = 0; }
#endif
        probe_blas();
        G.ok = true;
    } catch (const std::exception& e) {
        why = e.what();
        return false;
    }
    char b[512];
    std::snprintf(b, sizeof b, "%s, %d SMs; cuBLAS for fp32%s%s%s", G.name.c_str(), G.sms, G.use_h ? ", fp16" : "",
                  G.use_i8 ? ", int8" : "", G.use_sb ? ", batched" : "");
    describe = b;
    return true;
}

// Device properties as JSON members, for the tracer's topology record.
void gpu_topology_json(std::string& out) {
    out.clear();
    if (!G.ok) return;
    static const struct { cudaDeviceAttr a; const char* key; } attrs[] = {
        {cudaDevAttrMultiProcessorCount, "sm_count"},
        {cudaDevAttrMaxThreadsPerMultiProcessor, "max_threads_per_sm"},
        {cudaDevAttrMaxRegistersPerMultiprocessor, "registers_per_sm"},
        {cudaDevAttrMaxSharedMemoryPerMultiprocessor, "shared_memory_bytes_per_sm"},
        {cudaDevAttrMaxThreadsPerBlock, "max_threads_per_block"},
        {cudaDevAttrMaxRegistersPerBlock, "max_registers_per_block"},
        {cudaDevAttrMaxSharedMemoryPerBlock, "max_shared_memory_bytes_per_block"},
        {cudaDevAttrWarpSize, "warp_size"},
        {cudaDevAttrL2CacheSize, "l2_cache_bytes"},
        {cudaDevAttrClockRate, "clock_khz"},
        {cudaDevAttrMemoryClockRate, "memory_clock_khz"},
        {cudaDevAttrGlobalMemoryBusWidth, "memory_bus_bits"},
        {cudaDevAttrComputeCapabilityMajor, "compute_capability_major"},
        {cudaDevAttrComputeCapabilityMinor, "compute_capability_minor"},
        {cudaDevAttrIntegrated, "integrated"},
    };
    char b[160];
    std::snprintf(b, sizeof b, "\"name\":\"%s\",\"threads_per_block_used\":%d", G.name.c_str(), (int)TPB);
    out = b;
    for (const auto& x : attrs) {
        int v = 0;
        if (cudaDeviceGetAttribute(&v, x.a, 0) != cudaSuccess) { cudaGetLastError(); continue; }
        std::snprintf(b, sizeof b, ",\"%s\":%d", x.key, v);
        out += b;
    }
}

// A host thread other than the one that initialised CUDA binds the device before
// its first call (which may be a cuBLAS call rather than a runtime call).
void gpu_thread_init() {
    if (G.ok) { cudaSetDevice(0); cudaGetLastError(); }
}

// The tracer adjusts how much per-block detail it keeps.
void gpu_trace_config(long detail_blocks) {
#ifdef GB10_TRACE
    g_kt.detail = detail_blocks;
#else
    (void)detail_blocks;
#endif
}
