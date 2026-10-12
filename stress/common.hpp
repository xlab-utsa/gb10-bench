// common.hpp - shared definitions for the gb10_stress suite: precisions,
// workloads, problem shapes, and the interface every device backend implements.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---- precisions ------------------------------------------------------------
// fp32  float storage, float math
// fp16  half storage; values are widened to float for the math (cuBLAS runs the
//       matmul natively in half on the GPU)
// int8  8-bit storage, integer multiply-accumulate into int32; a real value is
//       raw/127. Non-linear steps (exp) are done in float on the scaled value.
// int4  4-bit storage, two values per byte, range [-8, 7], real value raw/8.
//       Weights are unpacked to int8 on the fly, then follow the int8 path.
enum Prec { F32 = 0, F16, I8, I4, NPREC };
static const char* const PREC_NAME[NPREC] = {"fp32", "fp16", "int8", "int4"};

inline size_t store_bytes(Prec p, size_t elems) {
    return p == F32 ? elems * 4 : p == F16 ? elems * 2 : p == I8 ? elems : (elems + 1) / 2;
}
inline float int_scale(Prec p) { return p == I8 ? 1.0f / 127.0f : p == I4 ? 1.0f / 8.0f : 1.0f; }
// Activations that meet 4-bit weights are kept in 8 bits, as quantised models do.
inline Prec act_prec(Prec p) { return p == I4 ? I8 : p; }

// ---- workloads, in rising order of complexity --------------------------------
enum TaskId { T_SUM = 0, T_HIST, T_STENCIL, T_SOFTMAX, T_LINEAR, T_CONV, T_ATTN, T_FLASH1, T_FLASH2, NTASK };
static const char* const TASK_NAME[NTASK] = {"sum",    "histogram", "stencil",  "softmax", "linear",
                                             "conv2d", "attention", "flash_v1", "flash_v2"};

enum { SUM_UNIT = 4096 };              // sum/histogram: elements per unit of work
static const float LOGIT_GAIN = 8.0f;  // softmax: logit = real value * 8
enum { FLASH_MAXD = 128 };             // largest head dimension the flash kernels accept

struct Shape { long long a, b, c, d; };  // meaning depends on the workload, see shape_str()

struct Operand {
    const char* name;
    Prec store;       // storage format
    size_t per_unit;  // elements per unit, or 0 when the operand is shared by all units
    size_t shared;    // elements of a shared operand
    uint64_t seed;
    bool positive;    // float data in [0, 1) instead of [-1, 1)
};

struct Case {
    TaskId task;
    Prec prec;
    Shape s;
    long long iters;  // passes over the data inside one timed run
    long long units;  // independent pieces of work; a CPU+GPU run splits these
    double work;      // operations per pass (flop-equivalents)
    double elements;  // input elements
    int nops;
    Operand op[3];

    size_t op_elems(int i) const { return op[i].shared ? op[i].shared : op[i].per_unit * (size_t)units; }
    size_t staging_bytes() const {
        size_t b = 0;
        for (int i = 0; i < nops; ++i) b += store_bytes(op[i].store, op_elems(i));
        return b;
    }
    std::string shape_str() const {
        char b[160];
        switch (task) {
            case T_SUM: case T_HIST: std::snprintf(b, sizeof b, "n=%lld", s.a); break;
            case T_STENCIL: std::snprintf(b, sizeof b, "grids=%lld N=%lld steps=%lld", s.a, s.b, s.c); break;
            case T_SOFTMAX: std::snprintf(b, sizeof b, "rows=%lld vocab=%lld", s.a, s.b); break;
            case T_LINEAR: std::snprintf(b, sizeof b, "tokens=%lld in=%lld out=%lld mats=%lld", s.a, s.b, s.c, s.d); break;
            case T_CONV: std::snprintf(b, sizeof b, "batch=%lld ch=%lld hw=%lld", s.a, s.b, s.c); break;
            default: std::snprintf(b, sizeof b, "seqs=%lld L=%lld d=%lld", s.a, s.b, s.c); break;
        }
        return b;
    }
};

inline Case make_case(TaskId t, Prec p, Shape s, long long iters) {
    Case c;
    c.task = t; c.prec = p; c.s = s; c.iters = iters; c.nops = 0;
    auto add = [&](const char* n, Prec st, size_t per_unit, size_t shared, uint64_t seed, bool pos = false) {
        c.op[c.nops++] = Operand{n, st, per_unit, shared, seed, pos};
    };
    const double a = (double)s.a, b = (double)s.b, cc = (double)s.c, d = (double)s.d;
    switch (t) {
        case T_SUM: case T_HIST:
            c.units = s.a / SUM_UNIT;
            add("x", p, SUM_UNIT, 0, 11, t == T_SUM);
            c.work = a; c.elements = a;
            break;
        case T_STENCIL:  // 5-point Jacobi on `a` independent N x N grids, c steps
            c.units = s.a;
            add("grid", p, (size_t)(s.b * s.b), 0, 21);
            c.work = 6.0 * a * b * b * cc; c.elements = a * b * b;
            break;
        case T_SOFTMAX:  // `a` rows of `b` logits
            c.units = s.a;
            add("logits", p, (size_t)s.b, 0, 31);
            c.work = 4.0 * a * b; c.elements = a * b;
            break;
        case T_LINEAR:  // Y = X * W_l^T for d weight matrices of c x b, on a tokens
            c.units = s.a;
            add("x", act_prec(p), (size_t)s.b, 0, 41);
            add("w", p, 0, (size_t)(s.d * s.c * s.b), 42);
            c.work = 2.0 * a * b * cc * d; c.elements = d * cc * b + a * b;
            break;
        case T_CONV:  // 3x3, same padding, b -> b channels, on a images of c x c
            c.units = s.a;
            add("x", act_prec(p), (size_t)(s.b * s.c * s.c), 0, 51);
            add("w", p, 0, (size_t)(s.b * s.b * 9), 52);
            c.work = 2.0 * a * b * b * 9.0 * cc * cc; c.elements = a * b * cc * cc;
            break;
        default:  // attention: a sequences (batch x heads) of length b, head dimension c
            c.units = s.a;
            add("q", p, (size_t)(s.b * s.c), 0, 61);
            add("k", p, (size_t)(s.b * s.c), 0, 62);
            add("v", p, (size_t)(s.b * s.c), 0, 63);
            c.work = a * (4.0 * b * b * cc + 4.0 * b * b); c.elements = 3.0 * a * b * cc;
            break;
    }
    return c;
}

// Problem sizes per workload, smallest first. The largest ones are the scale of
// a 7B-parameter instruct model: 4096-wide layers, a 32000-entry vocabulary,
// 128-dimensional attention heads, and about 6.6 billion weights in `linear`.
inline std::vector<Shape> task_levels(TaskId t, bool quick) {
    std::vector<Shape> v;
    switch (t) {
        case T_SUM: case T_HIST:
            for (int e : {18, 21, 24, 27, 30, 33}) v.push_back({1LL << e, 0, 0, 0});
            break;
        case T_STENCIL:
            v = {{4, 256, 8, 0}, {8, 1024, 8, 0}, {8, 4096, 8, 0}, {8, 8192, 16, 0}, {16, 16384, 16, 0}, {32, 16384, 16, 0}};
            break;
        case T_SOFTMAX:
            v = {{64, 1000, 0, 0}, {512, 8192, 0, 0}, {4096, 32000, 0, 0}, {32768, 32000, 0, 0}, {131072, 32000, 0, 0}};
            break;
        case T_LINEAR:
            v = {{64, 256, 512, 4}, {256, 1024, 2048, 8}, {512, 2048, 5504, 16}, {1024, 4096, 11008, 24},
                 {1024, 4096, 11008, 96}, {2048, 4096, 14336, 112}};
            break;
        case T_CONV:
            v = {{4, 16, 32, 0}, {8, 64, 64, 0}, {16, 128, 128, 0}, {32, 256, 128, 0}, {64, 512, 128, 0}, {64, 512, 256, 0}};
            break;
        default:
            v = {{8, 256, 64, 0}, {16, 1024, 64, 0}, {64, 2048, 128, 0}, {256, 4096, 128, 0}, {256, 8192, 128, 0},
                 {256, 16384, 128, 0}, {256, 32768, 128, 0}};
            break;
    }
    if (quick) v.resize(2);
    return v;
}

// ---- staging: the data a case starts from -----------------------------------
// One host buffer per operand, already in its storage format. Loading a case
// means copying from here into freshly allocated device memory.
struct Staging { void* p[3] = {nullptr, nullptr, nullptr}; };

// ---- device backends ----------------------------------------------------------
// One object per (case, device). load() and pass() are the two timed phases;
// pass() returns only when its work is finished. Any failure is thrown.
struct Backend {
    virtual ~Backend() {}
    virtual const char* impl() const = 0;                        // e.g. "openblas", "cublas", "kernel"
    virtual size_t bytes(long long units) const = 0;             // device memory needed for that many units
    virtual void load(const Staging& s, long long u0, long long u1) = 0;  // allocate, copy units [u0, u1) in
    virtual void pass() = 0;                                     // one pass over the loaded units
    virtual double checksum() = 0;                               // untimed digest of the result
    virtual void unload() = 0;                                   // free everything; safe after a failed load
};

Backend* make_cpu_backend(const Case& c);
Backend* make_gpu_backend(const Case& c);
std::string cpu_describe();                      // CPU matmul library in use
void cpu_set_threads(int n);                     // thread count for the CPU backends from now on
bool gpu_startup(std::string& describe, std::string& why);  // false when no GPU can be used
void gpu_topology_json(std::string& out);        // device properties, for the tracer
void gpu_trace_config(long detail_blocks);       // tracer only: per-block detail threshold
void gpu_thread_init();                          // call first on any extra host thread that will use the GPU

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] inline void fail(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    throw std::runtime_error(b);
}
