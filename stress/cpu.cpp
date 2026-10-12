// cpu.cpp - CPU backends for the gb10_stress suite (OpenMP, plus BLAS sgemm
// when linked). One class per workload; each handles all four precisions.
#include "common.hpp"
#include "trace.hpp"

#include <algorithm>
#include <omp.h>
#ifdef __linux__
#include <sys/mman.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__F16C__)
#include <immintrin.h>
#endif

typedef _Float16 f16;

// ============================================================================
// Memory and small helpers
// ============================================================================

static int NT() { return omp_get_max_threads(); }

static void* xalloc(size_t bytes) {
    const size_t align = 2u << 20;  // 2 MB, so the kernel can use huge pages
    const size_t n = (std::max<size_t>(bytes, 1) + align - 1) / align * align;
    void* p = nullptr;
    if (posix_memalign(&p, align, n) != 0) fail("host allocation of %.2f GiB failed", n / 1073741824.0);
#ifdef __linux__
    madvise(p, n, MADV_HUGEPAGE);
#endif
    TR_EVENT("host.alloc", tr::now_ns(), 0, "\"bytes\":%zu,\"addr\":\"%p\",\"tid\":%d,\"cpu\":%d", n, p, tr::cur_tid(), tr::cur_cpu());
    return p;
}

static void xfree(void* p) {
    if (!p) return;
    TR_EVENT("host.free", tr::now_ns(), 0, "\"addr\":\"%p\"", p);
    std::free(p);
}

template <class T> static T* xnew(size_t count) { return (T*)xalloc(count * sizeof(T)); }

// Parallel copy; the first touch of `dst` happens here, on the threads that copy.
static void par_copy(void* dst, const void* src, size_t bytes) {
    const size_t chunk = 4u << 20;
    const long n = (long)((bytes + chunk - 1) / chunk);
    #pragma omp parallel for schedule(dynamic, 4)
    for (long i = 0; i < n; ++i) {
        const size_t o = (size_t)i * chunk, len = std::min(chunk, bytes - o);
        TR_ITEM("host.copy", i, len, len);
        std::memcpy((char*)dst + o, (const char*)src + o, len);
    }
}

template <class F> static void par_blocks(size_t n, size_t block, const char* name, F&& f) {
    const long nb = (long)((n + block - 1) / block);
    #pragma omp parallel for schedule(dynamic, 4)
    for (long i = 0; i < nb; ++i) {
        const size_t o = (size_t)i * block, len = std::min(block, n - o);
        TR_ITEM(name, i, len, len);
        (void)name;
        f(o, len);
    }
}

// ---- storage conversions ------------------------------------------------------
static inline int8_t nib_lo(uint8_t b) { return (int8_t)((int8_t)(b << 4) >> 4); }
static inline int8_t nib_hi(uint8_t b) { return (int8_t)((int8_t)b >> 4); }

static void unpack4(const uint8_t* s, int8_t* d, size_t nbytes) {  // 2 values per byte
    for (size_t i = 0; i < nbytes; ++i) { d[2 * i] = nib_lo(s[i]); d[2 * i + 1] = nib_hi(s[i]); }
}
static void pack4(const int8_t* s, uint8_t* d, size_t nbytes) {
    for (size_t i = 0; i < nbytes; ++i) d[i] = (uint8_t)((s[2 * i] & 15) | ((s[2 * i + 1] & 15) << 4));
}
static void h2f(const f16* s, float* d, size_t n) { for (size_t i = 0; i < n; ++i) d[i] = (float)s[i]; }
// float -> half. GCC 13 does not vectorise this narrowing on its own, so the
// conversion instruction is used directly where the target has one.
static void f2h(const float* s, f16* d, size_t n) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= n; i += 4) vst1_f16((float16_t*)(d + i), vcvt_f16_f32(vld1q_f32(s + i)));
#elif defined(__F16C__)
    for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i*)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(s + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
    for (; i < n; ++i) d[i] = (f16)s[i];
}
static void i8f(const int8_t* s, float* d, size_t n) { for (size_t i = 0; i < n; ++i) d[i] = (float)s[i]; }

// Raw values of elements [off, off+n) as float (integers are not rescaled).
// For int4, off and n must be even.
static void to_float(Prec p, const void* src, size_t off, size_t n, float* dst) {
    switch (p) {
        case F32: std::memcpy(dst, (const float*)src + off, n * sizeof(float)); break;
        case F16: h2f((const f16*)src + off, dst, n); break;
        case I8: i8f((const int8_t*)src + off, dst, n); break;
        default: {
            const uint8_t* s = (const uint8_t*)src + off / 2;
            for (size_t i = 0; i < n / 2; ++i) { dst[2 * i] = (float)nib_lo(s[i]); dst[2 * i + 1] = (float)nib_hi(s[i]); }
        }
    }
}
// Same, as int8 (int8 and int4 only).
static void to_i8(Prec p, const void* src, size_t off, size_t n, int8_t* dst) {
    if (p == I8) std::memcpy(dst, (const int8_t*)src + off, n);
    else unpack4((const uint8_t*)src + off / 2, dst, n / 2);
}

// ---- vectorisable exp: Cephes polynomial, relative error about 1e-7 ------------
static inline float exp_ps(float x) {
    x = std::fmin(std::fmax(x, -87.0f), 88.0f);  // fmax/fmin vectorise on aarch64 where a ?: chain does not
    const float fx = std::floor(x * 1.44269504088896341f + 0.5f);
    float r = x - fx * 0.693359375f;
    r = r - fx * -2.12194440e-4f;
    const float z = r * r;
    float y = 1.9875691500E-4f;
    y = y * r + 1.3981999507E-3f;
    y = y * r + 8.3334519073E-3f;
    y = y * r + 4.1665795894E-2f;
    y = y * r + 1.6666665459E-1f;
    y = y * r + 5.0000001201E-1f;
    y = y * z + r + 1.0f;
    const uint32_t bits = (uint32_t)((int32_t)fx + 127) << 23;
    float p;
    std::memcpy(&p, &bits, sizeof p);
    return y * p;
}

// Softmax of one row; `t` and `out` may be the same array.
static void softmax_row(const float* t, int n, float* out) {
    float mx = t[0];
    #pragma omp simd reduction(max : mx)
    for (int i = 0; i < n; ++i) mx = t[i] > mx ? t[i] : mx;
    float s = 0.0f;
    #pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; ++i) { const float e = exp_ps(t[i] - mx); out[i] = e; s += e; }
    const float inv = 1.0f / s;
    #pragma omp simd
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

// ---- digests ------------------------------------------------------------------
template <class T> static double digest(const T* x, size_t n, bool square) {
    double total = 0.0;
    const size_t chunk = 8192;
    const long nch = (long)((n + chunk - 1) / chunk);
    #pragma omp parallel for schedule(dynamic, 64) reduction(+ : total)
    for (long c = 0; c < nch; ++c) {
        const size_t i0 = (size_t)c * chunk, i1 = std::min(n, i0 + chunk);
        double s = 0.0;
        for (size_t i = i0; i < i1; ++i) { const double v = (double)x[i]; s += square ? v * v : std::fabs(v); }
        total += s;
    }
    return total;
}
static double digest_store(Prec p, const void* x, size_t n) {  // sum of |raw value|
    if (p == F32) return digest((const float*)x, n, false);
    if (p == F16) return digest((const f16*)x, n, false);
    if (p == I8) return digest((const int8_t*)x, n, false);
    double total = 0.0;
    const uint8_t* b = (const uint8_t*)x;
    #pragma omp parallel for schedule(static) reduction(+ : total)
    for (long i = 0; i < (long)(n / 2); ++i) total += std::abs((int)nib_lo(b[i])) + std::abs((int)nib_hi(b[i]));
    return total;
}

// ============================================================================
// Matrix products
// ============================================================================

#ifdef USE_BLAS
extern "C" void sgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k, const float* alpha,
                       const float* a, const int* lda, const float* b, const int* ldb, const float* beta, float* c,
                       const int* ldc);
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
extern "C" char* openblas_get_config(void) __attribute__((weak));
extern "C" char* openblas_get_corename(void) __attribute__((weak));
#endif

std::string cpu_describe() {
#ifdef USE_BLAS
    if (!openblas_get_config) return "BLAS sgemm";
    char name[64] = "", ver[64] = "";
    std::sscanf(openblas_get_config(), "%63s %63s", name, ver);
    std::string s = std::string(name) + " " + ver;
    if (openblas_get_corename) s += std::string(" (") + openblas_get_corename() + " kernels)";
    return s;
#else
    return "built-in kernels (no BLAS)";
#endif
}

static void blas_threads(int n) {
#ifdef USE_BLAS
    if (openblas_set_num_threads) openblas_set_num_threads(n);
#else
    (void)n;
#endif
}

void cpu_set_threads(int n) {
    omp_set_num_threads(n);
    blas_threads(n);
}

// Column-major C = alpha * op(A) * op(B) + beta * C for op(A) in {N, T}, op(B) = N.
// Without BLAS this is a plain single-threaded reference.
static void gemm(char ta, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb,
                 float beta, float* C, int ldc) {
#ifdef USE_BLAS
    sgemm_(&ta, "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
#else
    for (int j = 0; j < n; ++j) {
        float* c = C + (size_t)j * ldc;
        const float* b = B + (size_t)j * ldb;
        if (beta == 0.0f) for (int i = 0; i < m; ++i) c[i] = 0.0f;
        else if (beta != 1.0f) for (int i = 0; i < m; ++i) c[i] *= beta;
        if (ta == 'N') {
            for (int p = 0; p < k; ++p) {
                const float s = alpha * b[p];
                const float* a = A + (size_t)p * lda;
                #pragma omp simd
                for (int i = 0; i < m; ++i) c[i] += s * a[i];
            }
        } else {
            for (int i = 0; i < m; ++i) {
                const float* a = A + (size_t)i * lda;
                float s = 0.0f;
                #pragma omp simd reduction(+ : s)
                for (int p = 0; p < k; ++p) s += a[p] * b[p];
                c[i] += alpha * s;
            }
        }
    }
#endif
}

// y[m] (+)= dot(x, W[m]) for rows m in [m0, m1) of W, four rows at a time so the
// four sums run as independent chains.
template <class E, class A>
static void dot_rows(const E* x, const E* W, A* y, int K, int m0, int m1, bool acc) {
    int m = m0;
    for (; m + 4 <= m1; m += 4) {
        const E *w0 = W + (size_t)m * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        A s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        #pragma omp simd reduction(+ : s0, s1, s2, s3)
        for (int k = 0; k < K; ++k) {
            const A xv = (A)x[k];
            s0 += xv * (A)w0[k]; s1 += xv * (A)w1[k]; s2 += xv * (A)w2[k]; s3 += xv * (A)w3[k];
        }
        if (acc) { y[m] += s0; y[m + 1] += s1; y[m + 2] += s2; y[m + 3] += s3; }
        else { y[m] = s0; y[m + 1] = s1; y[m + 2] = s2; y[m + 3] = s3; }
    }
    for (; m < m1; ++m) {
        const E* w = W + (size_t)m * K;
        A s = 0;
        #pragma omp simd reduction(+ : s)
        for (int k = 0; k < K; ++k) s += (A)x[k] * (A)w[k];
        if (acc) y[m] += s; else y[m] = s;
    }
}

// The int8 case is written without the simd pragma and without hoisting x[k]:
// in exactly this form GCC recognises four widening dot products and emits one
// SDOT per 16 products on the GB10 (with the pragma it falls back to SMULL).
static void dot_rows(const int8_t* x, const int8_t* W, int32_t* y, int K, int m0, int m1, bool acc) {
    int m = m0;
    for (; m + 4 <= m1; m += 4) {
        const int8_t *w0 = W + (size_t)m * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int k = 0; k < K; ++k) { s0 += x[k] * w0[k]; s1 += x[k] * w1[k]; s2 += x[k] * w2[k]; s3 += x[k] * w3[k]; }
        if (acc) { y[m] += s0; y[m + 1] += s1; y[m + 2] += s2; y[m + 3] += s3; }
        else { y[m] = s0; y[m + 1] = s1; y[m + 2] = s2; y[m + 3] = s3; }
    }
    for (; m < m1; ++m) {
        const int8_t* w = W + (size_t)m * K;
        int32_t s = 0;
        for (int k = 0; k < K; ++k) s += x[k] * w[k];
        if (acc) y[m] += s; else y[m] = s;
    }
}

// Y[t][m] (+)= dot(X[t], W[m]); X is T x K, W is M x K, Y is T x M, all row-major.
template <class E, class A>
static void dot_gemm_st(long long T, int K, int M, const E* X, const E* W, A* Y, bool acc) {
    for (long long t = 0; t < T; ++t) dot_rows(X + t * K, W, Y + t * M, K, 0, M, acc);
}
template <class E, class A>
static void dot_gemm(long long T, int K, int M, const E* X, const E* W, A* Y, bool acc) {
    const int MB = 16;  // rows of W per block: the block stays in L2 while X streams past
    const long nblk = (M + MB - 1) / MB;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long blk = 0; blk < nblk; ++blk) {
        const int m0 = (int)blk * MB, m1 = std::min(M, m0 + MB);
        TR_ITEM("linear.dot", blk, (size_t)T * K * sizeof(E) + (size_t)(m1 - m0) * K * sizeof(E), (size_t)T * (m1 - m0) * sizeof(A));
        for (long long t = 0; t < T; ++t) dot_rows(X + t * K, W, Y + t * M, K, m0, m1, acc);
    }
}

// ============================================================================
// Backends
// ============================================================================

struct CpuBase : Backend {
    Case c;
    long long U = 0;  // units loaded
    explicit CpuBase(const Case& cc) : c(cc) {}
    const char* impl() const override { return "openmp"; }

    // Working copy of an operand: units [u0, u1) of a per-unit one, all of a shared one.
    void* take(const Staging& s, int op, long long u0, long long u1) const {
        const Operand& o = c.op[op];
        const size_t elems = o.shared ? o.shared : o.per_unit * (size_t)(u1 - u0);
        const size_t off = o.shared ? 0 : store_bytes(o.store, o.per_unit * (size_t)u0);
        const size_t bytes = store_bytes(o.store, elems);
        void* p = xalloc(bytes);
        par_copy(p, (const char*)s.p[op] + off, bytes);
        return p;
    }
    size_t op_bytes(int op, long long units) const {
        const Operand& o = c.op[op];
        return store_bytes(o.store, o.shared ? o.shared : o.per_unit * (size_t)units);
    }
};

// ---- 1. sum --------------------------------------------------------------------
template <class T, class A> static double chunk_sum(const T* x, size_t len) {
    enum { L = 32 };  // independent partial sums, so the adds do not form one chain
    A acc[L] = {0};
    size_t i = 0;
    for (; i + L <= len; i += L)
        for (int j = 0; j < L; ++j) acc[j] += (A)x[i + j];
    for (int w = L / 2; w >= 1; w /= 2)
        for (int j = 0; j < w; ++j) acc[j] += acc[j + w];
    A s = acc[0];
    for (; i < len; ++i) s += (A)x[i];
    return (double)s;
}

struct CpuSum : CpuBase {
    void* x = nullptr;
    double result = 0.0;
    using CpuBase::CpuBase;
    size_t bytes(long long units) const override { return op_bytes(0, units); }
    void load(const Staging& s, long long u0, long long u1) override { U = u1 - u0; x = take(s, 0, u0, u1); }
    void unload() override { xfree(x); x = nullptr; }
    double checksum() override { return result; }
    void pass() override {
        const size_t n = (size_t)U * SUM_UNIT, CH = 8192;
        const long nch = (long)((n + CH - 1) / CH);
        const Prec p = c.prec;
        double total = 0.0;
        #pragma omp parallel for schedule(dynamic, 16) reduction(+ : total)
        for (long ch = 0; ch < nch; ++ch) {
            const size_t i0 = (size_t)ch * CH, len = std::min(CH, n - i0);
            TR_ITEM("sum", ch, store_bytes(p, len), 0);
            if (p == F32) total += chunk_sum<float, float>((const float*)x + i0, len);
            else if (p == F16) total += chunk_sum<f16, float>((const f16*)x + i0, len);
            else if (p == I8) total += chunk_sum<int8_t, int32_t>((const int8_t*)x + i0, len);
            else {
                int8_t tmp[8192];
                unpack4((const uint8_t*)x + i0 / 2, tmp, len / 2);
                total += chunk_sum<int8_t, int32_t>(tmp, len);
            }
        }
        result = total;
    }
};

// ---- 2. histogram ----------------------------------------------------------------
// 256 bins over [-1, 1) for the float formats, one bin per value for int8 (256)
// and int4 (16).
struct CpuHist : CpuBase {
    void* x = nullptr;
    uint64_t bins[256];
    using CpuBase::CpuBase;
    size_t bytes(long long units) const override { return op_bytes(0, units); }
    void load(const Staging& s, long long u0, long long u1) override { U = u1 - u0; x = take(s, 0, u0, u1); }
    void unload() override { xfree(x); x = nullptr; }
    double checksum() override {
        double s = 0.0;
        for (int i = 0; i < 256; ++i) s += (double)bins[i] * (i + 1);
        return s;
    }
    void pass() override {
        const size_t n = (size_t)U * SUM_UNIT, CH = 32768;
        const long nch = (long)((n + CH - 1) / CH);
        const Prec p = c.prec;
        std::memset(bins, 0, sizeof bins);
        #pragma omp parallel
        {
            uint64_t loc[256] = {0};
            uint8_t* idx = new uint8_t[CH];
            #pragma omp for schedule(dynamic, 4) nowait
            for (long ch = 0; ch < nch; ++ch) {
                const size_t i0 = (size_t)ch * CH, len = std::min(CH, n - i0);
                TR_ITEM("histogram", ch, store_bytes(p, len), 0);
                uint32_t t[4][256];
                std::memset(t, 0, sizeof t);
                if (p == I4) {
                    const uint8_t* b = (const uint8_t*)x + i0 / 2;
                    for (size_t i = 0; i < len / 2; ++i) { ++t[0][(b[i] & 15) ^ 8]; ++t[1][(b[i] >> 4) ^ 8]; }
                } else {
                    const uint8_t* q = idx;
                    if (p == I8) q = (const uint8_t*)x + i0;  // bin = value + 128, applied below
                    else if (p == F32) {
                        const float* v = (const float*)x + i0;
                        for (size_t i = 0; i < len; ++i) {
                            int b = (int)((v[i] + 1.0f) * 128.0f);
                            idx[i] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
                        }
                    } else {
                        const f16* v = (const f16*)x + i0;
                        for (size_t i = 0; i < len; ++i) {
                            int b = (int)(((float)v[i] + 1.0f) * 128.0f);
                            idx[i] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
                        }
                    }
                    const uint8_t flip = p == I8 ? 0x80 : 0;
                    size_t i = 0;
                    for (; i + 4 <= len; i += 4) {
                        ++t[0][q[i] ^ flip]; ++t[1][q[i + 1] ^ flip]; ++t[2][q[i + 2] ^ flip]; ++t[3][q[i + 3] ^ flip];
                    }
                    for (; i < len; ++i) ++t[0][q[i] ^ flip];
                }
                for (int b = 0; b < 256; ++b) loc[b] += (uint64_t)t[0][b] + t[1][b] + t[2][b] + t[3][b];
            }
            #pragma omp critical
            for (int b = 0; b < 256; ++b) bins[b] += loc[b];
            delete[] idx;
        }
    }
};

// ---- 3. stencil ------------------------------------------------------------------
// 5-point Jacobi with clamped edges. Float: 0.5*c + 0.125*(n+s+w+e).
// Integer: (4*c + n+s+w+e) >> 3, which stays inside the storage range.
// Half rows: the arithmetic runs on a float scratch row (the widening loads
// vectorise), and the result is narrowed in one pass.
static void stencil_row_h(const f16* n, const f16* c, const f16* s, f16* o, float* t, int N) {
    t[0] = 0.5f * (float)c[0] + 0.125f * ((float)n[0] + (float)s[0] + (float)c[0] + (float)c[1]);
    #pragma omp simd
    for (int x = 1; x < N - 1; ++x) t[x] = 0.5f * (float)c[x] + 0.125f * ((float)n[x] + (float)s[x] + (float)c[x - 1] + (float)c[x + 1]);
    t[N - 1] = 0.5f * (float)c[N - 1] + 0.125f * ((float)n[N - 1] + (float)s[N - 1] + (float)c[N - 2] + (float)c[N - 1]);
    f2h(t, o, N);
}
template <class T> static void stencil_row_f(const T* n, const T* c, const T* s, T* o, int N) {
    o[0] = (T)(0.5f * (float)c[0] + 0.125f * ((float)n[0] + (float)s[0] + (float)c[0] + (float)c[1]));
    #pragma omp simd
    for (int x = 1; x < N - 1; ++x)
        o[x] = (T)(0.5f * (float)c[x] + 0.125f * ((float)n[x] + (float)s[x] + (float)c[x - 1] + (float)c[x + 1]));
    o[N - 1] = (T)(0.5f * (float)c[N - 1] + 0.125f * ((float)n[N - 1] + (float)s[N - 1] + (float)c[N - 2] + (float)c[N - 1]));
}
static void stencil_row_i(const int8_t* n, const int8_t* c, const int8_t* s, int8_t* o, int N) {
    o[0] = (int8_t)((4 * c[0] + n[0] + s[0] + c[0] + c[1]) >> 3);
    #pragma omp simd
    for (int x = 1; x < N - 1; ++x) o[x] = (int8_t)((4 * c[x] + n[x] + s[x] + c[x - 1] + c[x + 1]) >> 3);
    o[N - 1] = (int8_t)((4 * c[N - 1] + n[N - 1] + s[N - 1] + c[N - 2] + c[N - 1]) >> 3);
}

struct CpuStencil : CpuBase {
    void *cur = nullptr, *nxt = nullptr;
    using CpuBase::CpuBase;
    size_t bytes(long long units) const override { return 2 * op_bytes(0, units); }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        cur = take(s, 0, u0, u1);
        nxt = xalloc(op_bytes(0, U));
    }
    void unload() override { xfree(cur); xfree(nxt); cur = nxt = nullptr; }
    double checksum() override { return digest_store(c.prec, cur, (size_t)U * c.s.b * c.s.b); }
    void pass() override {
        const int N = (int)c.s.b, steps = (int)c.s.c;
        const Prec p = c.prec;
        const long long rows = U * N;
        const size_t rb = store_bytes(p, N);
        for (int step = 0; step < steps; ++step) {
            const char *in = (const char*)cur;
            char* out = (char*)nxt;
            #pragma omp parallel
            {
                int8_t* tmp = p == I4 ? new int8_t[4 * (size_t)N] : nullptr;
                float* ftmp = p == F16 ? new float[N] : nullptr;
                #pragma omp for schedule(dynamic, 8)
                for (long long r = 0; r < rows; ++r) {
                    const long long g = r / N;
                    const int y = (int)(r % N);
                    TR_ITEM("stencil", r, 3 * rb, rb);
                    const size_t base = (size_t)g * N;  // first row of this grid
                    const char* rc = in + (base + y) * rb;
                    const char* rn = in + (base + (y > 0 ? y - 1 : 0)) * rb;
                    const char* rs = in + (base + (y < N - 1 ? y + 1 : N - 1)) * rb;
                    char* ro = out + (base + y) * rb;
                    if (p == F32) stencil_row_f((const float*)rn, (const float*)rc, (const float*)rs, (float*)ro, N);
                    else if (p == F16) stencil_row_h((const f16*)rn, (const f16*)rc, (const f16*)rs, (f16*)ro, ftmp, N);
                    else if (p == I8) stencil_row_i((const int8_t*)rn, (const int8_t*)rc, (const int8_t*)rs, (int8_t*)ro, N);
                    else {
                        unpack4((const uint8_t*)rn, tmp, rb);
                        unpack4((const uint8_t*)rc, tmp + N, rb);
                        unpack4((const uint8_t*)rs, tmp + 2 * (size_t)N, rb);
                        stencil_row_i(tmp, tmp + N, tmp + 2 * (size_t)N, tmp + 3 * (size_t)N, N);
                        pack4(tmp + 3 * (size_t)N, (uint8_t*)ro, rb);
                    }
                }
                delete[] tmp;
                delete[] ftmp;
            }
            std::swap(cur, nxt);
        }
    }
};

// ---- 4. softmax --------------------------------------------------------------------
struct CpuSoftmax : CpuBase {
    void* x = nullptr;
    float* out = nullptr;
    using CpuBase::CpuBase;
    size_t bytes(long long units) const override { return op_bytes(0, units) + (size_t)units * c.s.b * 4; }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        x = take(s, 0, u0, u1);
        out = xnew<float>((size_t)U * c.s.b);
    }
    void unload() override { xfree(x); xfree(out); x = nullptr; out = nullptr; }
    double checksum() override { return digest(out, (size_t)U * c.s.b, true); }
    void pass() override {
        const int V = (int)c.s.b;
        const Prec p = c.prec;
        const float gain = LOGIT_GAIN * int_scale(p);
        #pragma omp parallel
        {
            float* t = new float[V];
            #pragma omp for schedule(dynamic, 1)
            for (long long r = 0; r < U; ++r) {
                TR_ITEM("softmax", r, store_bytes(p, V), (size_t)V * 4);
                to_float(p, x, (size_t)r * V, V, t);
                #pragma omp simd
                for (int i = 0; i < V; ++i) t[i] *= gain;
                softmax_row(t, V, out + (size_t)r * V);
            }
            delete[] t;
        }
    }
};

// ---- 5. linear ---------------------------------------------------------------------
// Y[tokens x out] = sum over the weight matrices of X * W_l^T, i.e. the same
// tokens pushed through every matrix of the stack.
struct CpuLinear : CpuBase {
    void *X = nullptr, *W = nullptr, *Y = nullptr;
    float *Xf = nullptr, *Wf = nullptr;
    int8_t* W8 = nullptr;
    using CpuBase::CpuBase;
    bool is_int() const { return c.prec >= I8; }
    const char* impl() const override {
#ifdef USE_BLAS
        return is_int() ? "openmp" : "blas";
#else
        return "openmp";
#endif
    }
    size_t bytes(long long units) const override {
        const size_t K = c.s.b, M = c.s.c;
        size_t b = op_bytes(0, units) + op_bytes(1, units) + (size_t)units * M * 4;
        if (c.prec == F16) b += ((size_t)units * K + M * K) * 4;
        if (c.prec == I4) b += M * K;
        return b;
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        const size_t K = c.s.b, M = c.s.c;
        X = take(s, 0, u0, u1);
        W = take(s, 1, u0, u1);
        Y = xalloc((size_t)U * M * 4);
        if (c.prec == F16) { Xf = xnew<float>((size_t)U * K); Wf = xnew<float>(M * K); }
        if (c.prec == I4) W8 = xnew<int8_t>(M * K);
    }
    void unload() override {
        xfree(X); xfree(W); xfree(Y); xfree(Xf); xfree(Wf); xfree(W8);
        X = W = Y = nullptr; Xf = Wf = nullptr; W8 = nullptr;
    }
    double checksum() override {
        const size_t n = (size_t)U * c.s.c;
        return is_int() ? digest((const int32_t*)Y, n, false) : digest((const float*)Y, n, false);
    }
    void mm_f(const float* x, const float* w, bool acc) {
        const int K = (int)c.s.b, M = (int)c.s.c;
#ifdef USE_BLAS
        TR_SCOPE("cpu.blas", "\"routine\":\"sgemm\",\"m\":%d,\"n\":%lld,\"k\":%d,\"tid\":%d,\"cpu\":%d", M, U, K, tr::cur_tid(), tr::cur_cpu());
        gemm('T', M, (int)U, K, 1.0f, w, K, x, K, acc ? 1.0f : 0.0f, (float*)Y, M);
#else
        dot_gemm<float, float>(U, K, M, x, w, (float*)Y, acc);
#endif
    }
    void pass() override {
        const size_t K = c.s.b, M = c.s.c;
        const int mats = (int)c.s.d;
        const Prec p = c.prec;
        if (p == F16) par_blocks((size_t)U * K, 1 << 16, "linear.widen", [&](size_t o, size_t n) { h2f((const f16*)X + o, Xf + o, n); });
        for (int l = 0; l < mats; ++l) {
            const char* w = (const char*)W + (size_t)l * store_bytes(p, M * K);
            if (p == F32) mm_f((const float*)X, (const float*)w, l > 0);
            else if (p == F16) {
                par_blocks(M * K, 1 << 16, "linear.widen", [&](size_t o, size_t n) { h2f((const f16*)w + o, Wf + o, n); });
                mm_f(Xf, Wf, l > 0);
            } else if (p == I8) dot_gemm<int8_t, int32_t>(U, (int)K, (int)M, (const int8_t*)X, (const int8_t*)w, (int32_t*)Y, l > 0);
            else {
                par_blocks(M * K / 2, 1 << 16, "linear.unpack", [&](size_t o, size_t n) { unpack4((const uint8_t*)w + o, W8 + 2 * o, n); });
                dot_gemm<int8_t, int32_t>(U, (int)K, (int)M, (const int8_t*)X, W8, (int32_t*)Y, l > 0);
            }
        }
    }
};

// ---- 6. conv2d ---------------------------------------------------------------------
// Direct 3x3 convolution with zero padding. Three taps of one kernel row are
// applied in one pass over the output row.
template <class E, class A>
static void conv_run(const E* X, const A* Wc, A* Out, long long B, int C, int H, int Wd) {
    const long long rows = B * H;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long long r = 0; r < rows; ++r) {
        const long long b = r / H;
        const int y = (int)(r % H);
        TR_ITEM("conv2d", r, (size_t)3 * C * Wd * sizeof(E), (size_t)C * Wd * sizeof(A));
        for (int co = 0; co < C; ++co) {
            A* o = Out + ((size_t)(b * C + co) * H + y) * Wd;
            for (int x = 0; x < Wd; ++x) o[x] = 0;
            for (int ci = 0; ci < C; ++ci) {
                const A* w = Wc + ((size_t)co * C + ci) * 9;
                for (int ky = 0; ky < 3; ++ky) {
                    const int yy = y + ky - 1;
                    if (yy < 0 || yy >= H) continue;
                    const E* in = X + ((size_t)(b * C + ci) * H + yy) * Wd;
                    const A w0 = w[ky * 3], w1 = w[ky * 3 + 1], w2 = w[ky * 3 + 2];
                    o[0] += w1 * (A)in[0] + w2 * (A)in[1];
                    #pragma omp simd
                    for (int x = 1; x < Wd - 1; ++x) o[x] += w0 * (A)in[x - 1] + w1 * (A)in[x] + w2 * (A)in[x + 1];
                    o[Wd - 1] += w0 * (A)in[Wd - 2] + w1 * (A)in[Wd - 1];
                }
            }
        }
    }
}

struct CpuConv : CpuBase {
    void *X = nullptr, *Out = nullptr, *Wc = nullptr;
    using CpuBase::CpuBase;
    bool is_int() const { return c.prec >= I8; }
    size_t nw() const { return (size_t)c.s.b * c.s.b * 9; }
    size_t bytes(long long units) const override {
        return op_bytes(0, units) + (size_t)units * c.s.b * c.s.c * c.s.c * 4 + nw() * 4;
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        X = take(s, 0, u0, u1);
        Out = xalloc((size_t)U * c.s.b * c.s.c * c.s.c * 4);
        Wc = xalloc(nw() * 4);  // weights widened once to the accumulator type
        if (is_int()) {
            std::vector<int8_t> w8(nw() + 1);
            to_i8(c.prec, s.p[1], 0, nw() + (nw() & 1), w8.data());
            for (size_t i = 0; i < nw(); ++i) ((int32_t*)Wc)[i] = w8[i];
        } else to_float(c.prec, s.p[1], 0, nw(), (float*)Wc);
    }
    void unload() override { xfree(X); xfree(Out); xfree(Wc); X = Out = Wc = nullptr; }
    double checksum() override {
        const size_t n = (size_t)U * c.s.b * c.s.c * c.s.c;
        return is_int() ? digest((const int32_t*)Out, n, false) : digest((const float*)Out, n, false);
    }
    void pass() override {
        const int C = (int)c.s.b, HW = (int)c.s.c;
        if (c.prec == F32) conv_run<float, float>((const float*)X, (const float*)Wc, (float*)Out, U, C, HW, HW);
        else if (c.prec == F16) conv_run<f16, float>((const f16*)X, (const float*)Wc, (float*)Out, U, C, HW, HW);
        else conv_run<int8_t, int32_t>((const int8_t*)X, (const int32_t*)Wc, (int32_t*)Out, U, C, HW, HW);
    }
};

// ---- 7. attention, scores materialised -----------------------------------------------
// O = softmax(Q K^T / sqrt(d)) V per sequence, with the full L x L score matrix
// held in memory. Q, K and V are widened to float; the products are sgemm calls.
struct CpuAttn : CpuBase {
    void *Q = nullptr, *K = nullptr, *V = nullptr;
    float* O = nullptr;
    std::vector<float*> S, tf;  // per slot: scores, and widened q|k|v
    using CpuBase::CpuBase;
    int slots(long long units) const { return units >= NT() ? NT() : 1; }
    size_t bytes(long long units) const override {
        const size_t L = c.s.b, d = c.s.c;
        return 3 * op_bytes(0, units) + (size_t)units * L * d * 4 +
               (size_t)slots(units) * (L * L * 4 + (c.prec != F32 ? 3 * L * d * 4 : 0));
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        const size_t L = c.s.b, d = c.s.c;
        Q = take(s, 0, u0, u1); K = take(s, 1, u0, u1); V = take(s, 2, u0, u1);
        O = xnew<float>((size_t)U * L * d);
        for (int i = 0; i < slots(U); ++i) {
            S.push_back(xnew<float>(L * L));
            tf.push_back(c.prec != F32 ? xnew<float>(3 * L * d) : nullptr);
        }
    }
    void unload() override {
        xfree(Q); xfree(K); xfree(V); xfree(O);
        for (float* p : S) xfree(p);
        for (float* p : tf) xfree(p);
        S.clear(); tf.clear();
        Q = K = V = nullptr; O = nullptr;
    }
    double checksum() override { return digest(O, (size_t)U * c.s.b * c.s.c, false); }
    void one(long long u, int slot, bool inner_parallel) {
        const int L = (int)c.s.b, d = (int)c.s.c;
        const size_t ld = (size_t)L * d;
        const float sI = int_scale(c.prec), alpha = sI * sI / std::sqrt((float)d);
        TR_ITEM("attention.seq", u, 3 * store_bytes(c.prec, ld), ld * 4 + (size_t)L * L * 4);
        const float *q, *k, *v;
        if (c.prec == F32) { q = (const float*)Q + u * ld; k = (const float*)K + u * ld; v = (const float*)V + u * ld; }
        else {
            float* t = tf[slot];
            to_float(c.prec, Q, u * ld, ld, t);
            to_float(c.prec, K, u * ld, ld, t + ld);
            to_float(c.prec, V, u * ld, ld, t + 2 * ld);
            q = t; k = t + ld; v = t + 2 * ld;
        }
        float* s = S[slot];
        gemm('T', L, L, d, alpha, k, d, q, d, 0.0f, s, L);  // s[i*L + j] = alpha * q_i . k_j
        #pragma omp parallel for schedule(dynamic, 16) if (inner_parallel)
        for (int i = 0; i < L; ++i) softmax_row(s + (size_t)i * L, L, s + (size_t)i * L);
        gemm('N', d, L, L, sI, v, d, s, L, 0.0f, O + u * ld, d);  // o_i = sum_j p_ij v_j
    }
    void pass() override {
        if (slots(U) > 1) {
            blas_threads(1);  // one sequence per thread, each with single-threaded BLAS
            #pragma omp parallel for schedule(dynamic, 1)
            for (long long u = 0; u < U; ++u) one(u, omp_get_thread_num(), false);
            blas_threads(NT());
        } else {
            for (long long u = 0; u < U; ++u) one(u, 0, true);
        }
    }
};

// ---- 8, 9. FlashAttention v1 and v2 ---------------------------------------------------
// Same result as the attention above, but the score matrix is never stored:
// queries and keys are processed in tiles with a running row maximum m and row
// sum l (online softmax).
//   v1: key/value tiles in the outer loop. The running state m, l and the
//       normalised output are kept in memory and reloaded, renormalised and
//       written back for every query tile at every step.
//   v2: query tiles in the outer loop, each independent. The state stays in
//       per-thread scratch, the output is accumulated unnormalised and divided
//       by l once at the end.
// Integer formats compute q.k as int8 dot products; V is widened to float.
struct CpuFlash : CpuBase {
    enum { BR = 128, BC = 128 };
    struct Scr { float *qf, *kf, *vf, *S, *oacc, *m, *l; int8_t *q8, *k8; int32_t* Si; };
    void *Q = nullptr, *K = nullptr, *V = nullptr;
    float *O = nullptr, *Mx = nullptr, *Ls = nullptr;
    std::vector<Scr> scr;
    int version;
    CpuFlash(const Case& cc, int v) : CpuBase(cc), version(v) {}
    bool is_int() const { return c.prec >= I8; }
    size_t scr_bytes() const {
        const size_t d = c.s.c;
        return (size_t)NT() * ((3 * BR * d + BR * BC + BR * d + 2 * BR) * 4 + 2 * BR * d + BR * BC * 4);
    }
    size_t bytes(long long units) const override {
        const size_t L = c.s.b, d = c.s.c;
        return 3 * op_bytes(0, units) + (size_t)units * L * d * 4 + (version == 1 ? (size_t)units * L * 8 : 0) + scr_bytes();
    }
    void load(const Staging& s, long long u0, long long u1) override {
        U = u1 - u0;
        const size_t L = c.s.b, d = c.s.c;
        if (d > FLASH_MAXD) fail("flash attention supports head dimensions up to %d", (int)FLASH_MAXD);
        Q = take(s, 0, u0, u1); K = take(s, 1, u0, u1); V = take(s, 2, u0, u1);
        O = xnew<float>((size_t)U * L * d);
        if (version == 1) { Mx = xnew<float>((size_t)U * L); Ls = xnew<float>((size_t)U * L); }
        for (int i = 0; i < NT(); ++i) {
            Scr z;
            z.qf = xnew<float>(BR * d); z.kf = xnew<float>(BC * d); z.vf = xnew<float>(BC * d);
            z.S = xnew<float>(BR * BC); z.oacc = xnew<float>(BR * d); z.m = xnew<float>(BR); z.l = xnew<float>(BR);
            z.q8 = xnew<int8_t>(BR * d); z.k8 = xnew<int8_t>(BC * d); z.Si = xnew<int32_t>(BR * BC);
            scr.push_back(z);
        }
    }
    void unload() override {
        xfree(Q); xfree(K); xfree(V); xfree(O); xfree(Mx); xfree(Ls);
        for (Scr& z : scr) {
            xfree(z.qf); xfree(z.kf); xfree(z.vf); xfree(z.S); xfree(z.oacc); xfree(z.m); xfree(z.l);
            xfree(z.q8); xfree(z.k8); xfree(z.Si);
        }
        scr.clear();
        Q = K = V = nullptr; O = Mx = Ls = nullptr;
    }
    double checksum() override { return digest(O, (size_t)U * c.s.b * c.s.c, false); }

    // Rows [r0, r0+n) of one sequence's operand, widened for the score product.
    void tile(const void* src, long long u, int r0, int n, float* f, int8_t* i8, const float*& fo, const int8_t*& io) const {
        const size_t d = c.s.c, off = ((size_t)u * c.s.b + r0) * d;
        fo = nullptr; io = nullptr;
        if (c.prec == F32) fo = (const float*)src + off;
        else if (c.prec == F16) { h2f((const f16*)src + off, f, n * d); fo = f; }
        else if (c.prec == I8) io = (const int8_t*)src + off;
        else { unpack4((const uint8_t*)src + off / 2, i8, n * d / 2); io = i8; }
    }
    // z.S[i*bc + j] = alpha * q_i . k_j
    void scores(Scr& z, const float* qf, const int8_t* q8, const float* kf, const int8_t* k8, int br, int bc) const {
        const int d = (int)c.s.c;
        const float sI = int_scale(c.prec), alpha = sI * sI / std::sqrt((float)d);
        if (!is_int()) gemm('T', bc, br, d, alpha, kf, d, qf, d, 0.0f, z.S, bc);
        else {
            dot_gemm_st<int8_t, int32_t>(br, d, bc, q8, k8, z.Si, false);
            const int n = br * bc;
            #pragma omp simd
            for (int i = 0; i < n; ++i) z.S[i] = (float)z.Si[i] * alpha;
        }
    }

    void pass() override {
        const int L = (int)c.s.b, d = (int)c.s.c;
        const int ntile = (L + BR - 1) / BR;
        const long long total = U * ntile;
        const float sI = int_scale(c.prec);
        const size_t tb = store_bytes(c.prec, (size_t)BR * d);
        (void)tb;
        blas_threads(1);
        if (version == 2) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (long long w = 0; w < total; ++w) {
                const long long u = w / ntile;
                const int i0 = (int)(w % ntile) * BR, br = std::min<int>(BR, L - i0);
                TR_ITEM("flash_v2.qtile", w, tb + 2 * store_bytes(c.prec, (size_t)L * d), (size_t)br * d * 4);
                Scr& z = scr[omp_get_thread_num()];
                const float *qf, *kf; const int8_t *q8, *k8;
                tile(Q, u, i0, br, z.qf, z.q8, qf, q8);
                for (int i = 0; i < br; ++i) { z.m[i] = -1e30f; z.l[i] = 0.0f; }
                std::memset(z.oacc, 0, (size_t)br * d * sizeof(float));
                for (int j0 = 0; j0 < L; j0 += BC) {
                    const int bc = std::min<int>(BC, L - j0);
                    tile(K, u, j0, bc, z.kf, z.k8, kf, k8);
                    scores(z, qf, q8, kf, k8, br, bc);
                    for (int i = 0; i < br; ++i) {
                        float* row = z.S + (size_t)i * bc;
                        float mt = row[0];
                        #pragma omp simd reduction(max : mt)
                        for (int j = 0; j < bc; ++j) mt = row[j] > mt ? row[j] : mt;
                        if (mt > z.m[i]) {  // new maximum: rescale what has been accumulated
                            const float a = exp_ps(z.m[i] - mt);
                            z.l[i] *= a;
                            float* o = z.oacc + (size_t)i * d;
                            #pragma omp simd
                            for (int cc = 0; cc < d; ++cc) o[cc] *= a;
                            z.m[i] = mt;
                        }
                        const float m = z.m[i];
                        float sum = 0.0f;
                        #pragma omp simd reduction(+ : sum)
                        for (int j = 0; j < bc; ++j) { const float e = exp_ps(row[j] - m); row[j] = e; sum += e; }
                        z.l[i] += sum;
                    }
                    to_float(c.prec, V, ((size_t)u * L + j0) * d, (size_t)bc * d, z.vf);
                    gemm('N', d, br, bc, 1.0f, z.vf, d, z.S, bc, 1.0f, z.oacc, d);
                }
                for (int i = 0; i < br; ++i) {
                    const float inv = sI / z.l[i];
                    float* o = O + ((size_t)u * L + i0 + i) * d;
                    const float* a = z.oacc + (size_t)i * d;
                    #pragma omp simd
                    for (int cc = 0; cc < d; ++cc) o[cc] = a[cc] * inv;
                }
            }
        } else {
            const size_t n = (size_t)U * L;
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)n; ++i) { Mx[i] = -1e30f; Ls[i] = 0.0f; }
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < (long long)(n * d); ++i) O[i] = 0.0f;
            for (int j0 = 0; j0 < L; j0 += BC) {
                const int bc = std::min<int>(BC, L - j0);
                #pragma omp parallel for schedule(dynamic, 1)
                for (long long w = 0; w < total; ++w) {
                    const long long u = w / ntile;
                    const int i0 = (int)(w % ntile) * BR, br = std::min<int>(BR, L - i0);
                    TR_ITEM("flash_v1.step", w, 3 * tb + (size_t)br * d * 4 + br * 8, (size_t)br * d * 4 + br * 8);
                    Scr& z = scr[omp_get_thread_num()];
                    const float *qf, *kf; const int8_t *q8, *k8;
                    tile(Q, u, i0, br, z.qf, z.q8, qf, q8);
                    tile(K, u, j0, bc, z.kf, z.k8, kf, k8);
                    scores(z, qf, q8, kf, k8, br, bc);
                    float* Ot = O + ((size_t)u * L + i0) * d;
                    for (int i = 0; i < br; ++i) {
                        const size_t idx = (size_t)u * L + i0 + i;
                        const float m_old = Mx[idx], l_old = Ls[idx];
                        float* row = z.S + (size_t)i * bc;
                        float mt = row[0];
                        #pragma omp simd reduction(max : mt)
                        for (int j = 0; j < bc; ++j) mt = row[j] > mt ? row[j] : mt;
                        const float m_new = mt > m_old ? mt : m_old;
                        const float a = exp_ps(m_old - m_new);
                        float sum = 0.0f;
                        #pragma omp simd reduction(+ : sum)
                        for (int j = 0; j < bc; ++j) { const float e = exp_ps(row[j] - m_new); row[j] = e; sum += e; }
                        const float l_new = a * l_old + sum, f = l_old * a;
                        float* o = Ot + (size_t)i * d;
                        #pragma omp simd
                        for (int cc = 0; cc < d; ++cc) o[cc] *= f;  // back to the unnormalised sum
                        Mx[idx] = m_new; Ls[idx] = l_new; z.l[i] = l_new;
                    }
                    to_float(c.prec, V, ((size_t)u * L + j0) * d, (size_t)bc * d, z.vf);
                    gemm('N', d, br, bc, sI, z.vf, d, z.S, bc, 1.0f, Ot, d);
                    for (int i = 0; i < br; ++i) {
                        const float inv = 1.0f / z.l[i];
                        float* o = Ot + (size_t)i * d;
                        #pragma omp simd
                        for (int cc = 0; cc < d; ++cc) o[cc] *= inv;  // renormalise, as v1 does every step
                    }
                }
            }
        }
        blas_threads(NT());
    }
};

Backend* make_cpu_backend(const Case& c) {
    switch (c.task) {
        case T_SUM: return new CpuSum(c);
        case T_HIST: return new CpuHist(c);
        case T_STENCIL: return new CpuStencil(c);
        case T_SOFTMAX: return new CpuSoftmax(c);
        case T_LINEAR: return new CpuLinear(c);
        case T_CONV: return new CpuConv(c);
        case T_ATTN: return new CpuAttn(c);
        case T_FLASH1: return new CpuFlash(c, 1);
        default: return new CpuFlash(c, 2);
    }
}
