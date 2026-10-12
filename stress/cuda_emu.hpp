// cuda_emu.hpp - a stand-in for the CUDA runtime and cuBLAS so that gpu.cu can
// be compiled by a plain C++ compiler and run on the CPU (make EMULATE=1).
// It exists to check the GPU code path on a machine without a GPU: every
// kernel in gpu.cu runs unchanged, one emulated thread at a time. It is slow
// and its timings mean nothing.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

struct dim3 { unsigned x, y, z; };
static thread_local dim3 threadIdx = {0, 0, 0}, blockIdx = {0, 0, 0};
static const dim3 blockDim = {256, 1, 1};

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
enum { cudaDeviceScheduleBlockingSync = 4 };
enum cudaDeviceAttr {
    cudaDevAttrMultiProcessorCount, cudaDevAttrMaxThreadsPerMultiProcessor, cudaDevAttrMaxRegistersPerMultiprocessor,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor, cudaDevAttrWarpSize, cudaDevAttrL2CacheSize, cudaDevAttrClockRate,
    cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor, cudaDevAttrIntegrated,
    cudaDevAttrGlobalMemoryBusWidth, cudaDevAttrMemoryClockRate, cudaDevAttrMaxThreadsPerBlock,
    cudaDevAttrMaxRegistersPerBlock, cudaDevAttrMaxSharedMemoryPerBlock
};
struct cudaDeviceProp { char name[256]; };
struct cudaFuncAttributes { int numRegs; size_t sharedSizeBytes, localSizeBytes, constSizeBytes; int maxThreadsPerBlock; };

enum { EMU_SMS = 8 };  // the emulated device pretends to have this many SMs

inline const char* cudaGetErrorString(cudaError_t) { return "emulated CUDA error"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { std::strcpy(p->name, "emulated GPU (CPU threads)"); return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
    switch (a) {
        case cudaDevAttrMultiProcessorCount: *v = EMU_SMS; break;
        case cudaDevAttrMaxThreadsPerMultiProcessor: *v = 2048; break;
        case cudaDevAttrMaxRegistersPerMultiprocessor: case cudaDevAttrMaxRegistersPerBlock: *v = 65536; break;
        case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *v = 102400; break;
        case cudaDevAttrMaxSharedMemoryPerBlock: *v = 49152; break;
        case cudaDevAttrWarpSize: *v = 32; break;
        case cudaDevAttrClockRate: *v = 1000000; break;  // kHz, so that one "cycle" is one nanosecond
        case cudaDevAttrMaxThreadsPerBlock: *v = 1024; break;
        case cudaDevAttrIntegrated: *v = 1; break;
        default: *v = 0;
    }
    return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { *p = (T*)std::malloc(n ? n : 1); return *p ? cudaSuccess : 2; }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
template <class F> inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, F) {
    a->numRegs = 32; a->sharedSizeBytes = 0; a->localSizeBytes = 0; a->constSizeBytes = 0; a->maxThreadsPerBlock = 1024;
    return cudaSuccess;
}

// ---- device-side functions ----------------------------------------------------
typedef _Float16 __half;
inline float __half2float(__half h) { return (float)h; }
inline __half __float2half(float f) { return (__half)f; }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline long long atomicMin(long long* p, long long v) {
    long long old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v < old && !__atomic_compare_exchange_n(p, &old, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return old;
}
inline long long atomicMax(long long* p, long long v) {
    long long old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v > old && !__atomic_compare_exchange_n(p, &old, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    return old;
}
inline long long clock64() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
inline unsigned kt_smid() { return (unsigned)omp_get_thread_num() % EMU_SMS; }

namespace emu {
// Run a kernel body for threads [0, ceil(n/256)*256), block by block.
template <class F> inline void launch(long long n, F&& body) {
    const long long blocks = (n + blockDim.x - 1) / blockDim.x;
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long b = 0; b < blocks; ++b) {
        blockIdx.x = (unsigned)b;
        for (unsigned t = 0; t < blockDim.x; ++t) { threadIdx.x = t; body(); }
    }
}
}  // namespace emu

// ---- cuBLAS, written straight from the BLAS definitions -------------------------
typedef int cublasHandle_t;
typedef int cublasStatus_t;
enum { CUBLAS_STATUS_SUCCESS = 0 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cudaDataType { CUDA_R_16F = 2, CUDA_R_32F = 0, CUDA_R_8I = 3, CUDA_R_32I = 10 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32I = 72 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };
inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = 1; return 0; }

namespace emu {
template <class TA, class TC, class S>
inline void gemm(int ta, int m, int n, int k, S alpha, const TA* A, int lda, const TA* B, int ldb, S beta, TC* C, int ldc) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            S acc = 0;
            for (int p = 0; p < k; ++p) acc += (S)(ta == CUBLAS_OP_N ? A[i + (size_t)p * lda] : A[p + (size_t)i * lda]) * (S)B[p + (size_t)j * ldb];
            TC& c = C[i + (size_t)j * ldc];
            c = (TC)(alpha * acc + (beta != 0 ? beta * (S)c : 0));
        }
}
}  // namespace emu

inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t, int m, int n, int k, const float* alpha,
                                  const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    emu::gemm<float, float, float>(ta, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    return 0;
}
inline cublasStatus_t cublasHgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t, int m, int n, int k, const __half* alpha,
                                  const __half* A, int lda, const __half* B, int ldb, const __half* beta, __half* C, int ldc) {
    emu::gemm<__half, __half, float>(ta, m, n, k, (float)*alpha, A, lda, B, ldb, (float)*beta, C, ldc);
    return 0;
}
inline cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t, int m, int n, int k, const void* alpha,
                                   const void* A, cudaDataType at, int lda, const void* B, cudaDataType, int ldb, const void* beta,
                                   void* C, cudaDataType ct, int ldc, cublasComputeType_t, cublasGemmAlgo_t) {
    if (at != CUDA_R_8I || ct != CUDA_R_32I) return 15;  // not supported
    emu::gemm<signed char, int, int>(ta, m, n, k, *(const int*)alpha, (const signed char*)A, lda, (const signed char*)B, ldb,
                                     *(const int*)beta, (int*)C, ldc);
    return 0;
}
inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                                const float* alpha, const float* A, int lda, long long sa, const float* B, int ldb,
                                                long long sb, const float* beta, float* C, int ldc, long long sc, int count) {
    for (int i = 0; i < count; ++i) cublasSgemm(h, ta, tb, m, n, k, alpha, A + i * sa, lda, B + i * sb, ldb, beta, C + i * sc, ldc);
    return 0;
}
