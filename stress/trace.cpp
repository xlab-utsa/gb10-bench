// trace.cpp - the event recorder behind gb10_trace (built only with -DGB10_TRACE).
//
// Events are buffered in memory per thread while a case runs and written, in
// time order, once the case has finished, so the log describes what happened
// without the file I/O sitting inside the phases it describes. The format is
// JSON Lines: one object per line, every one with
//     "t"    nanoseconds since the trace started
//     "dur"  duration in nanoseconds (0 for instantaneous events)
//     "ev"   event type
// plus the members documented in the README.
//
// A sampler thread records the state of the whole machine every few
// milliseconds: per-core load and frequency, memory, temperatures, and the GPU
// counters NVML exposes.
#include "trace.hpp"
#ifdef GB10_TRACE

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>

namespace tr {

struct Ev { uint64_t t, dur; std::string line; };

struct ThreadBuf {
    int tid = 0;
    std::mutex m;
    std::vector<Ev> evs;
    // open "cpu.work" span
    const char* k = nullptr;
    int cpu = -1;
    uint64_t t0 = 0, t1 = 0;
    long long first = 0, last = 0, count = 0;
    size_t r = 0, w = 0;
};

static std::chrono::steady_clock::time_point T0 = std::chrono::steady_clock::now();
static std::FILE* out = nullptr;
static uint64_t slice_ns = 1000000;
static std::mutex reg_m;
static std::vector<ThreadBuf*> reg;
static std::atomic<int> next_tid(0);
static std::atomic<uint64_t> written(0);
static std::atomic<int> ctx_case(-1), ctx_run(0);
static std::atomic<const char*> ctx_phase("");

void context(int case_id, int run, const char* phase) { ctx_case = case_id; ctx_run = run; ctx_phase = phase; }

static ThreadBuf& mine() {
    static thread_local ThreadBuf* tb = nullptr;
    if (!tb) {
        tb = new ThreadBuf;  // kept for the life of the process: commit() reads it after the thread ends
        tb->tid = next_tid++;
        std::lock_guard<std::mutex> g(reg_m);
        reg.push_back(tb);
    }
    return *tb;
}

uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - T0).count();
}
int cur_cpu() {
#ifdef __linux__
    return sched_getcpu();
#else
    return -1;
#endif
}
int cur_tid() { return mine().tid; }

static thread_local bool untagged = false;  // set while replaying records whose case is not the current one

static void push(ThreadBuf& tb, const char* ev, uint64_t t, uint64_t dur, const char* fields) {
    char head[96];
    std::snprintf(head, sizeof head, "{\"t\":%llu,\"dur\":%llu,\"ev\":\"%s\"", (unsigned long long)t, (unsigned long long)dur, ev);
    Ev e{t, dur, std::string(head)};
    const int cs = ctx_case;
    if (cs >= 0 && !untagged) {
        char c[96];
        std::snprintf(c, sizeof c, ",\"case\":%d,\"run\":%d,\"phase\":\"%s\"", cs, ctx_run.load(), ctx_phase.load());
        e.line += c;
    }
    if (fields && *fields) { e.line += ","; e.line += fields; }
    e.line += "}";
    std::lock_guard<std::mutex> g(tb.m);
    tb.evs.push_back(std::move(e));
}

void event(const char* ev, uint64_t t_ns, uint64_t dur_ns, const char* fmt, ...) {
    char b[4096];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    push(mine(), ev, t_ns, dur_ns, b);
}

Scope::Scope(const char* ev_, const char* fmt, ...) : ev(ev_), t0(now_ns()) {
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(fields, sizeof fields, fmt, ap);
    va_end(ap);
}
Scope::~Scope() { push(mine(), ev, t0, now_ns() - t0, fields); }

static void close_span(ThreadBuf& tb) {
    if (!tb.k) return;
    char b[320];
    std::snprintf(b, sizeof b, "\"tid\":%d,\"cpu\":%d,\"kernel\":\"%s\",\"first\":%lld,\"last\":%lld,\"items\":%lld,\"bytes_read\":%zu,\"bytes_written\":%zu",
                  tb.tid, tb.cpu, tb.k, tb.first, tb.last, tb.count, tb.r, tb.w);
    const char* k = tb.k;
    tb.k = nullptr;
    (void)k;
    push(tb, "cpu.work", tb.t0, tb.t1 - tb.t0, b);
}

// Merge consecutive items of one kernel on one thread and one core, up to the
// slice length, so that a loop of a million tiny iterations becomes a few
// hundred spans instead of a million events.
void item(const char* kernel, long long index, size_t bytes_r, size_t bytes_w, uint64_t t0, uint64_t t1) {
    ThreadBuf& tb = mine();
    const int cpu = cur_cpu();
    if (tb.k && (tb.k != kernel || tb.cpu != cpu || t1 - tb.t0 > slice_ns || t0 - tb.t1 > 200000)) close_span(tb);
    if (!tb.k) { tb.k = kernel; tb.cpu = cpu; tb.t0 = t0; tb.first = index; tb.count = 0; tb.r = tb.w = 0; }
    tb.t1 = t1; tb.last = index; ++tb.count; tb.r += bytes_r; tb.w += bytes_w;
}

void flush_items() {
    std::lock_guard<std::mutex> g(reg_m);
    for (ThreadBuf* tb : reg) close_span(*tb);
}

// ---- optional: CUPTI activity records --------------------------------------------
// Adds what the CUDA driver itself reports for every kernel (cuBLAS internals
// included), copy and memset: device-side start and end, grid, registers per
// thread. Fields are read by offset: the leading part of these records has had
// the same layout in every CUPTI release checked (kernel records 4-9, memcpy
// 1-5, memset 1-4), which avoids naming a versioned struct.
#ifdef HAVE_CUPTI
}  // namespace tr
#include <cupti.h>
namespace tr {
static int64_t cupti_shift = 0;  // our clock minus CUPTI's
template <class T> static T at(const uint8_t* p, size_t off) { T v; std::memcpy(&v, p + off, sizeof v); return v; }
static void CUPTIAPI cupti_request(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8u << 20;
    *buffer = (uint8_t*)std::malloc(*size);
    *max_records = 0;
}
static void CUPTIAPI cupti_complete(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
    // These arrive when the buffer is flushed, after the case that produced them:
    // they are written with their own timestamps and without a case tag.
    untagged = true;
    CUpti_Activity* rec = nullptr;
    while (valid > 0 && cuptiActivityGetNextRecord(buffer, valid, &rec) == CUPTI_SUCCESS) {
        const uint8_t* p = (const uint8_t*)rec;
        const uint64_t start = at<uint64_t>(p, 16), end = at<uint64_t>(p, 24);
        const uint64_t t = (uint64_t)((int64_t)start + cupti_shift);
        if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const char* name = at<const char*>(p, 104);
            std::string nm = name ? name : "?";
            for (char& ch : nm) if (ch == '"' || ch == '\\') ch = '\'';
            if (nm.size() > 300) nm.resize(300);
            event("cupti.kernel", t, end - start,
                  "\"name\":\"%s\",\"grid\":[%d,%d,%d],\"block\":[%d,%d,%d],\"regs_per_thread\":%u,\"static_shared_bytes\":%d,"
                  "\"dynamic_shared_bytes\":%d,\"local_bytes_per_thread\":%u,\"stream\":%u,\"correlation\":%u,\"queued_ns\":%lld,\"submitted_ns\":%lld",
                  nm.c_str(), at<int32_t>(p, 52), at<int32_t>(p, 56), at<int32_t>(p, 60), at<int32_t>(p, 64), at<int32_t>(p, 68), at<int32_t>(p, 72),
                  (unsigned)at<uint16_t>(p, 6), at<int32_t>(p, 76), at<int32_t>(p, 80), at<uint32_t>(p, 84), at<uint32_t>(p, 48), at<uint32_t>(p, 92),
                  (long long)((int64_t)at<uint64_t>(p, 120) + cupti_shift), (long long)((int64_t)at<uint64_t>(p, 128) + cupti_shift));
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            event("cupti.memcpy", t, end - start, "\"copy_kind\":%u,\"src_kind\":%u,\"dst_kind\":%u,\"bytes\":%llu,\"stream\":%u,\"correlation\":%u",
                  (unsigned)at<uint8_t>(p, 4), (unsigned)at<uint8_t>(p, 5), (unsigned)at<uint8_t>(p, 6), (unsigned long long)at<uint64_t>(p, 8),
                  at<uint32_t>(p, 40), at<uint32_t>(p, 44));
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            event("cupti.memset", t, end - start, "\"bytes\":%llu,\"stream\":%u,\"correlation\":%u", (unsigned long long)at<uint64_t>(p, 8),
                  at<uint32_t>(p, 40), at<uint32_t>(p, 44));
        }
    }
    untagged = false;
    std::free(buffer);
}
static bool cupti_on = false;
static void cupti_start() {
    if (cuptiActivityRegisterCallbacks(cupti_request, cupti_complete) != CUPTI_SUCCESS) return;
    uint64_t ct = 0;
    cuptiGetTimestamp(&ct);
    cupti_shift = (int64_t)now_ns() - (int64_t)ct;
    cupti_on = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS;
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
}
static void cupti_flush() { if (cupti_on) cuptiActivityFlushAll(0); }
#else
static bool cupti_on = false;
static void cupti_start() {}
static void cupti_flush() {}
#endif

// ---- machine description ---------------------------------------------------------
static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}
static std::string jstr(std::string s) {
    for (char& ch : s) if (ch == '"' || ch == '\\' || (unsigned char)ch < 32) ch = ' ';
    return s;
}
static int ncpu() { return (int)sysconf(_SC_NPROCESSORS_CONF); }

static const char* arm_core_name(unsigned implementer, unsigned part) {
    if (implementer != 0x41) return "";
    switch (part) {
        case 0xd85: return "Cortex-X925";
        case 0xd87: return "Cortex-A725";
        case 0xd84: return "Cortex-X4";
        case 0xd81: return "Cortex-A720";
        case 0xd4f: return "Neoverse-V2";
        default: return "";
    }
}

static void describe_machine(const std::string& cmdline, const std::string& gpu_json, const std::string& cpu_blas, int sample_ms) {
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    event("meta", 0, 0,
          "\"format\":\"gb10-trace\",\"version\":1,\"host\":\"%s\",\"command\":\"%s\",\"unix_time\":%lld,\"time_unit\":\"ns since trace start\","
          "\"sample_period_ms\":%d,\"cpu_span_slice_ns\":%llu,\"cupti\":%s,\"cpu_matmul\":\"%s\"",
          jstr(host).c_str(), jstr(cmdline).c_str(), (long long)time(nullptr), sample_ms, (unsigned long long)slice_ns, cupti_on ? "true" : "false",
          jstr(cpu_blas).c_str());
    for (int c = 0; c < ncpu(); ++c) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        const std::string midr_s = slurp(base + "/regs/identification/midr_el1");
        const unsigned long midr = midr_s.empty() ? 0 : std::strtoul(midr_s.c_str(), nullptr, 16);
        const unsigned impl = (unsigned)(midr >> 24) & 0xff, part = (unsigned)(midr >> 4) & 0xfff;
        std::string caches;
        for (int i = 0; i < 6; ++i) {
            const std::string cb = base + "/cache/index" + std::to_string(i);
            const std::string level = slurp(cb + "/level");
            if (level.empty()) break;
            caches += (caches.empty() ? "" : ",") + std::string("{\"level\":") + level + ",\"type\":\"" + jstr(slurp(cb + "/type")) + "\",\"size\":\"" +
                      jstr(slurp(cb + "/size")) + "\",\"shared_with\":\"" + jstr(slurp(cb + "/shared_cpu_list")) + "\"}";
        }
        const std::string cap = slurp(base + "/cpu_capacity"), fmax = slurp(base + "/cpufreq/cpuinfo_max_freq"),
                          fmin = slurp(base + "/cpufreq/cpuinfo_min_freq"), online = slurp(base + "/online");
        event("topology.cpu", 0, 0,
              "\"cpu\":%d,\"online\":%s,\"core\":\"%s\",\"midr\":\"%s\",\"capacity\":%s,\"max_khz\":%s,\"min_khz\":%s,\"cluster_cpus\":\"%s\",\"caches\":[%s]",
              c, online == "0" ? "false" : "true", arm_core_name(impl, part), jstr(midr_s).c_str(), cap.empty() ? "null" : cap.c_str(),
              fmax.empty() ? "null" : fmax.c_str(), fmin.empty() ? "null" : fmin.c_str(), jstr(slurp(base + "/topology/cluster_cpus_list")).c_str(),
              caches.c_str());
    }
    double total = 0, avail = 0, swap = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string k;
        double kb;
        while (f >> k >> kb) {
            if (k == "MemTotal:") total = kb * 1024;
            if (k == "MemAvailable:") avail = kb * 1024;
            if (k == "SwapTotal:") swap = kb * 1024;
            f.ignore(256, '\n');
        }
    }
    event("topology.memory", 0, 0, "\"total_bytes\":%.0f,\"available_bytes\":%.0f,\"swap_bytes\":%.0f,\"page_bytes\":%ld,\"transparent_hugepages\":\"%s\"",
          total, avail, swap, sysconf(_SC_PAGESIZE), jstr(slurp("/sys/kernel/mm/transparent_hugepage/enabled")).c_str());
    if (!gpu_json.empty()) event("topology.gpu", 0, 0, "%s", gpu_json.c_str());
    event("topology.link", 0, 0,
          "\"between\":[\"cpu\",\"gpu\"],\"shared_memory\":true,\"note\":\"CPU and GPU share one DRAM pool. gpu.xfer events are copies between "
          "host and device allocations inside that pool; gpu.kernel events carry estimated bytes read and written. Which physical link "
          "carries that traffic is not observable from software.\"");
}

// ---- sampler thread ----------------------------------------------------------------
struct Nvml {
    void* lib = nullptr;
    void* dev = nullptr;
    int (*util)(void*, unsigned*) = nullptr;       // nvmlUtilization_t {gpu, memory}
    int (*temp)(void*, int, unsigned*) = nullptr;
    int (*power)(void*, unsigned*) = nullptr;      // milliwatts
    int (*clock)(void*, int, unsigned*) = nullptr; // MHz
    int (*mem)(void*, unsigned long long*) = nullptr;  // nvmlMemory_t {total, free, used}
    void open() {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) return;
        auto init = (int (*)())dlsym(lib, "nvmlInit_v2");
        auto handle = (int (*)(unsigned, void**))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        if (!init || !handle || init() != 0 || handle(0, &dev) != 0) { dev = nullptr; return; }
        util = (int (*)(void*, unsigned*))dlsym(lib, "nvmlDeviceGetUtilizationRates");
        temp = (int (*)(void*, int, unsigned*))dlsym(lib, "nvmlDeviceGetTemperature");
        power = (int (*)(void*, unsigned*))dlsym(lib, "nvmlDeviceGetPowerUsage");
        clock = (int (*)(void*, int, unsigned*))dlsym(lib, "nvmlDeviceGetClockInfo");
        mem = (int (*)(void*, unsigned long long*))dlsym(lib, "nvmlDeviceGetMemoryInfo");
    }
};

static std::atomic<bool> sampling(false);
static std::thread sampler;

static void sample_loop(int period_ms) {
    Nvml nv;
    nv.open();
    const int n = ncpu();
    std::vector<unsigned long long> busy0(n, 0), tot0(n, 0);
    std::vector<std::string> zone_type;
    for (int z = 0; z < 64; ++z) {
        const std::string t = slurp("/sys/class/thermal/thermal_zone" + std::to_string(z) + "/type");
        if (t.empty()) break;
        zone_type.push_back(jstr(t));
    }
    {
        std::string zt;
        for (size_t i = 0; i < zone_type.size(); ++i) zt += (i ? ",\"" : "\"") + zone_type[i] + "\"";
        event("topology.sensors", 0, 0, "\"thermal_zones\":[%s],\"nvml\":%s", zt.c_str(), nv.dev ? "true" : "false");
    }
    bool first = true;
    while (sampling) {
        const uint64_t t = now_ns();
        std::string util = "[", freq = "[", temps = "[";
        {
            std::ifstream f("/proc/stat");
            std::string line;
            std::vector<double> u(n, -1.0);
            while (std::getline(f, line)) {
                if (line.compare(0, 3, "cpu") != 0 || line.size() < 4 || line[3] < '0' || line[3] > '9') continue;
                std::istringstream is(line.substr(3));
                int id;
                unsigned long long v[8] = {0};
                is >> id;
                for (auto& x : v) is >> x;
                if (id < 0 || id >= n) continue;
                const unsigned long long tot = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7], busy = tot - v[3] - v[4];
                if (!first && tot > tot0[id]) u[id] = 100.0 * (double)(busy - busy0[id]) / (double)(tot - tot0[id]);
                busy0[id] = busy; tot0[id] = tot;
            }
            char b[32];
            for (int c = 0; c < n; ++c) {
                if (u[c] < 0) std::snprintf(b, sizeof b, "%snull", c ? "," : "");
                else std::snprintf(b, sizeof b, "%s%.0f", c ? "," : "", u[c]);
                util += b;
                const std::string fk = slurp("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/scaling_cur_freq");
                freq += (c ? "," : "") + (fk.empty() ? std::string("null") : fk);
            }
        }
        for (size_t z = 0; z < zone_type.size(); ++z) {
            const std::string v = slurp("/sys/class/thermal/thermal_zone" + std::to_string(z) + "/temp");
            char b[32];
            if (v.empty()) std::snprintf(b, sizeof b, "%snull", z ? "," : "");
            else std::snprintf(b, sizeof b, "%s%.1f", z ? "," : "", std::atof(v.c_str()) / 1000.0);
            temps += b;
        }
        double avail = 0;
        {
            std::ifstream f("/proc/meminfo");
            std::string k;
            double kb;
            while (f >> k >> kb) { if (k == "MemAvailable:") { avail = kb * 1024; break; } f.ignore(256, '\n'); }
        }
        long rss_pages = 0;
        { std::ifstream f("/proc/self/statm"); long size; f >> size >> rss_pages; }
        std::string gpu;
        if (nv.dev) {
            unsigned u[2] = {0, 0}, tc = 0, mw = 0, mhz = 0;
            unsigned long long m[3] = {0, 0, 0};
            char b[96];
            if (nv.util && nv.util(nv.dev, u) == 0) { std::snprintf(b, sizeof b, ",\"gpu_util_pct\":%u,\"gpu_mem_util_pct\":%u", u[0], u[1]); gpu += b; }
            if (nv.temp && nv.temp(nv.dev, 0, &tc) == 0) { std::snprintf(b, sizeof b, ",\"gpu_temp_c\":%u", tc); gpu += b; }
            if (nv.power && nv.power(nv.dev, &mw) == 0) { std::snprintf(b, sizeof b, ",\"gpu_power_w\":%.1f", mw / 1000.0); gpu += b; }
            if (nv.clock && nv.clock(nv.dev, 0, &mhz) == 0) { std::snprintf(b, sizeof b, ",\"gpu_clock_mhz\":%u", mhz); gpu += b; }
            if (nv.mem && nv.mem(nv.dev, m) == 0) { std::snprintf(b, sizeof b, ",\"gpu_mem_used_bytes\":%llu", m[2]); gpu += b; }
        }
        event("sample", t, 0, "\"cpu_util_pct\":%s],\"cpu_khz\":%s],\"temps_c\":%s],\"mem_available_bytes\":%.0f,\"process_rss_bytes\":%.0f%s", util.c_str(),
              freq.c_str(), temps.c_str(), avail, (double)rss_pages * (double)sysconf(_SC_PAGESIZE), gpu.c_str());
        first = false;
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    }
}

// ---- lifecycle -------------------------------------------------------------------
void commit() {
    flush_items();
    cupti_flush();
    std::vector<Ev> all;
    {
        std::lock_guard<std::mutex> g(reg_m);
        for (ThreadBuf* tb : reg) {
            std::lock_guard<std::mutex> h(tb->m);
            for (Ev& e : tb->evs) all.push_back(std::move(e));
            tb->evs.clear();
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Ev& a, const Ev& b) { return a.t < b.t; });
    if (out) {
        for (const Ev& e : all) { std::fputs(e.line.c_str(), out); std::fputc('\n', out); }
        std::fflush(out);
    }
    written += all.size();
}

void start(const std::string& path, int sample_ms, int slice_us, const std::string& cmdline, const std::string& gpu_json,
           const std::string& cpu_blas) {
    out = std::fopen(path.c_str(), "w");
    if (!out) { std::fprintf(stderr, "error: cannot write %s\n", path.c_str()); std::exit(1); }
    slice_ns = (uint64_t)std::max(slice_us, 1) * 1000;
    cupti_start();
    describe_machine(cmdline, gpu_json, cpu_blas, sample_ms);
    commit();
    if (sample_ms > 0) {
        sampling = true;
        sampler = std::thread(sample_loop, sample_ms);
    }
}

void stop() {
    if (sampling) { sampling = false; sampler.join(); }
    commit();
    if (out) { std::fclose(out); out = nullptr; }
    std::printf("%llu events written\n", (unsigned long long)written.load());
}

}  // namespace tr
#endif  // GB10_TRACE
