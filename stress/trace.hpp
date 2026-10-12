// trace.hpp - tracing hooks. In the normal build every macro here expands to
// nothing, so the measured code carries no instrumentation at all. Built with
// -DGB10_TRACE (make trace), the same call sites record events (see trace.cpp).
#pragma once
#ifdef GB10_TRACE
#include <cstddef>
#include <cstdint>

namespace tr {
uint64_t now_ns();  // nanoseconds since the trace started
int cur_cpu();      // logical CPU the calling thread is on right now
int cur_tid();      // small sequential id of the calling thread

// One event. `fmt` produces extra JSON members, e.g. "\"bytes\":%zu".
void event(const char* ev, uint64_t t_ns, uint64_t dur_ns, const char* fmt, ...) __attribute__((format(printf, 4, 5)));

// A unit of CPU work (one loop iteration of a kernel). Consecutive items of one
// kernel on one thread and core are merged into a single "cpu.work" span.
void item(const char* kernel, long long index, size_t bytes_r, size_t bytes_w, uint64_t t0, uint64_t t1);
void flush_items();  // close the open spans of every thread; call outside parallel regions

// Every event recorded from now on is tagged with this case number, run and
// phase (case < 0: no tag). Run -1 is the warm-up run.
void context(int case_id, int run, const char* phase);

struct Item {
    const char* k; long long i; size_t r, w; uint64_t t0;
    Item(const char* k_, long long i_, size_t r_, size_t w_) : k(k_), i(i_), r(r_), w(w_), t0(now_ns()) {}
    ~Item() { item(k, i, r, w, t0, now_ns()); }
};

// An event whose duration is the lifetime of the object.
struct Scope {
    const char* ev; uint64_t t0; char fields[320];
    Scope(const char* ev_, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
    ~Scope();
};
}  // namespace tr

#define TR_CAT2(a, b) a##b
#define TR_CAT(a, b) TR_CAT2(a, b)
#define TR_ITEM(k, i, r, w) tr::Item TR_CAT(tr_item_, __LINE__)((k), (long long)(i), (size_t)(r), (size_t)(w))
#define TR_SCOPE(...) tr::Scope TR_CAT(tr_scope_, __LINE__)(__VA_ARGS__)
#define TR_EVENT(...) tr::event(__VA_ARGS__)
#define TR_CONTEXT(c, r, p) tr::context((c), (r), (p))
#else
#define TR_ITEM(k, i, r, w) ((void)0)
#define TR_SCOPE(...) ((void)0)
#define TR_EVENT(...) ((void)0)
#define TR_CONTEXT(c, r, p) ((void)0)
#endif

#ifdef GB10_TRACE
#define TR_ON 1
#else
#define TR_ON 0
#endif
