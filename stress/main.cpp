// main.cpp - the gb10_stress runner.
//
// For every workload x precision x device it climbs a ladder of problem sizes,
// then keeps the largest size and multiplies the number of passes, until the
// next case would no longer fit its time budget. A case is `--runs` complete
// runs (25 by default); each run is
//     load     allocate device memory and copy the operands in     (timed)
//     compute  `iters` passes of the workload                      (timed)
// and the CSV reports the mean, min, max and standard deviation of both and of
// their sum. Timing is two clock reads around each phase and nothing else:
// there is no instrumentation inside the kernels in this build.
//
// The same sources built with -DGB10_TRACE (make trace) give gb10_trace, which
// runs selected cases once and writes a detailed event log instead.
#include "common.hpp"
#include "trace.hpp"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <fstream>
#include <map>
#include <memory>
#include <omp.h>
#include <sstream>
#include <thread>
#include <unistd.h>
#ifdef __linux__
#include <dirent.h>
#include <sched.h>
#include <sys/mman.h>
#endif

int plot_stress(const std::string& csv_path, const std::string& prefix);
#ifdef GB10_TRACE
namespace tr {
void start(const std::string& path, int sample_ms, int slice_us, const std::string& cmdline, const std::string& gpu_json,
           const std::string& cpu_blas);
void commit();  // write the buffered events of the case that just finished, in time order
void stop();
}
#endif

enum Kind { CPU = 0, GPU = 1, BOTH = 2 };

// What a sweep runs on. On a machine with more than one kind of CPU core there
// is one CPU device per kind (threads confined to those cores) besides the
// device that uses all of them.
struct DevSpec {
    std::string name;       // "cpu:x925", "cpu:a725", "gpu", "cpu", "cpu+gpu"
    Kind kind;
    std::vector<int> cpus;  // logical CPUs the process may use during this sweep
    int threads;
    std::string what;       // for the log
};

struct Options {
    bool task[NTASK], prec[NPREC];
    std::string devices;          // empty: every device this machine offers
    std::string cpu_sets;         // "name=cpulist;name=cpulist": replaces the detected core types
    int runs = 25;
    int warmup = 1;               // untimed runs before the measured ones
    double case_budget = 1800.0;  // seconds for all runs of one case
    double mem_gb = 90.0;
    std::string out = "stress.csv";
    bool fresh = false, plot = true, quick = false, repeat = true;
    int max_level = 99, only_level = -1;
    // tracer
    std::string trace_out = "trace.jsonl";
    std::string shares = "stress.csv";  // benchmark results the tracer takes its cpu+gpu splits from
    int sample_ms = 20, slice_us = 1000;
    long block_detail = 512;
};

// ============================================================================
// Staging data
// ============================================================================
static float hash01(uint64_t i) {  // uniform [0, 1), reproducible for any thread count
    uint64_t z = i + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

static void* stage_alloc(size_t bytes) {
    const size_t align = 2u << 20, n = (std::max<size_t>(bytes, 1) + align - 1) / align * align;
    void* p = nullptr;
    if (posix_memalign(&p, align, n) != 0) fail("staging allocation of %.2f GiB failed", n / 1073741824.0);
#ifdef __linux__
    madvise(p, n, MADV_HUGEPAGE);
#endif
    return p;
}

static void generate(const Case& c, Staging& st) {
    for (int i = 0; i < c.nops; ++i) {
        const Operand& o = c.op[i];
        const long long n = (long long)c.op_elems(i);
        const uint64_t salt = o.seed * 0x100000001B3ull << 20;
        void* p = stage_alloc(store_bytes(o.store, (size_t)n));
        st.p[i] = p;
        if (o.store == F32) {
            #pragma omp parallel for schedule(static)
            for (long long j = 0; j < n; ++j) { const float u = hash01((uint64_t)j ^ salt); ((float*)p)[j] = o.positive ? u : 2.0f * u - 1.0f; }
        } else if (o.store == F16) {
            #pragma omp parallel for schedule(static)
            for (long long j = 0; j < n; ++j) { const float u = hash01((uint64_t)j ^ salt); ((_Float16*)p)[j] = (_Float16)(o.positive ? u : 2.0f * u - 1.0f); }
        } else if (o.store == I8) {
            #pragma omp parallel for schedule(static)
            for (long long j = 0; j < n; ++j) ((int8_t*)p)[j] = (int8_t)((int)(hash01((uint64_t)j ^ salt) * 255.0f) - 127);
        } else {
            #pragma omp parallel for schedule(static)
            for (long long j = 0; j < (n + 1) / 2; ++j) {
                const int lo = (int)(hash01((uint64_t)(2 * j) ^ salt) * 16.0f) & 15, hi = (int)(hash01((uint64_t)(2 * j + 1) ^ salt) * 16.0f) & 15;
                ((uint8_t*)p)[j] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

static void release(Staging& st) {
    for (void*& p : st.p) { std::free(p); p = nullptr; }
}

// ============================================================================
// System readings (taken between cases, never inside a timed phase)
// ============================================================================
static double meminfo_gib(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    double kb;
    while (f >> k >> kb) {
        if (k == key) return kb / 1048576.0;
        f.ignore(256, '\n');
    }
    return -1.0;
}

static double max_temp_c() {
    double best = -1.0;
    for (int z = 0; z < 32; ++z) {
        std::ifstream f("/sys/class/thermal/thermal_zone" + std::to_string(z) + "/temp");
        double v;
        if (f >> v) best = std::max(best, v / 1000.0);
    }
    return best;
}

static std::string fmt_time(double s) {
    char b[48];
    if (s < 1e-6) std::snprintf(b, sizeof b, "%.3g ns", s * 1e9);
    else if (s < 1e-3) std::snprintf(b, sizeof b, "%.3g us", s * 1e6);
    else if (s < 0.9995) std::snprintf(b, sizeof b, "%.3g ms", s * 1e3);
    else if (s < 600.0) std::snprintf(b, sizeof b, "%.3g s", s);
    else std::snprintf(b, sizeof b, "%.1f min", s / 60.0);
    return b;
}

// ============================================================================
// One case on one device
// ============================================================================
struct Stat {
    double mean = 0, mn = 0, mx = 0, sd = 0;
    static Stat of(const std::vector<double>& v) {
        Stat s;
        if (v.empty()) return s;
        s.mn = *std::min_element(v.begin(), v.end());
        s.mx = *std::max_element(v.begin(), v.end());
        for (double x : v) s.mean += x;
        s.mean /= v.size();
        for (double x : v) s.sd += (x - s.mean) * (x - s.mean);
        s.sd = v.size() > 1 ? std::sqrt(s.sd / (v.size() - 1)) : 0.0;
        return s;
    }
};

struct Result {
    bool ok = false, aborted = false;
    std::string err, impl;
    int runs = 0;
    Stat load, comp, total;
    double cpu_load = 0, gpu_load = 0, cpu_comp = 0, gpu_comp = 0;  // per-device means (cpu+gpu runs)
    double checksum = 0;
};

struct Part {
    std::unique_ptr<Backend> b;
    long long u0 = 0, u1 = 0;
    bool gpu = false;
    double t = 0;       // duration of the phase just run
    std::string err;
};

// Run `fn` on every part at once: the GPU part on its own thread, the CPU part
// here (its OpenMP team does the work). Returns the wall time of the phase.
template <class F> static double run_parts(std::vector<Part>& parts, F&& fn) {
    const double t0 = now_s();
    auto one = [&](Part& p) {
        const double s0 = now_s();
        TR_SCOPE("phase", "\"device\":\"%s\",\"units\":%lld,\"tid\":%d,\"cpu\":%d", p.gpu ? "gpu" : "cpu", p.u1 - p.u0, tr::cur_tid(), tr::cur_cpu());
        try { fn(p); } catch (const std::exception& e) { p.err = e.what(); }
        p.t = now_s() - s0;
    };
    if (parts.size() == 2) {
        Part *g = parts[0].gpu ? &parts[0] : &parts[1], *c = parts[0].gpu ? &parts[1] : &parts[0];
        std::thread tg([&] { gpu_thread_init(); one(*g); });
        one(*c);
        tg.join();
    } else one(parts[0]);
    const double dt = now_s() - t0;
#ifdef GB10_TRACE
    tr::flush_items();
#endif
    return dt;
}

static Result run_case(const Case& c, Kind dev, const Staging& st, long long ug, int runs, int warmup, double hard_limit, int case_id) {
    (void)case_id;
    Result r;
    std::vector<Part> parts;
    auto add = [&](bool gpu, long long u0, long long u1) {
        Part p;
        p.b.reset(gpu ? make_gpu_backend(c) : make_cpu_backend(c));
        p.u0 = u0; p.u1 = u1; p.gpu = gpu;
        parts.push_back(std::move(p));
    };
    if (dev == CPU) add(false, 0, c.units);
    else if (dev == GPU) add(true, 0, c.units);
    else { add(true, 0, ug); add(false, ug, c.units); }
    for (Part& p : parts) {
        if (!p.b) { r.err = "device unavailable"; return r; }
        r.impl += (r.impl.empty() ? "" : "+") + std::string(p.b->impl());
    }

    std::vector<double> load, comp, total, cl, gl, cc, gc;
    bool have_sum = false;
    std::atomic<bool> abort(false);
    const double case_t0 = now_s();
    auto first_error = [&]() { for (Part& p : parts) if (!p.err.empty()) return p.err; return std::string(); };

    // Runs before 0 are warm-up: identical, but not recorded. They absorb one-off
    // costs (thread pools starting, the first launch of a kernel, cold caches).
    for (int run = -warmup; run < runs && !abort; ++run) {
        TR_CONTEXT(case_id, run, "load");
        const double tl = run_parts(parts, [&](Part& p) { p.b->load(st, p.u0, p.u1); });
        std::string e = first_error();
        double pl[2] = {0, 0};
        for (Part& p : parts) pl[p.gpu] = p.t;
        double tc = 0;
        if (e.empty()) {
            TR_CONTEXT(case_id, run, "compute");
            tc = run_parts(parts, [&](Part& p) {
                for (long long it = 0; it < c.iters && !abort; ++it) {
                    p.b->pass();
                    if (it < c.iters - 1 && now_s() - case_t0 > hard_limit) abort = true;  // give up in the middle of a run
                }
            });
            e = first_error();
        }
        const bool over = now_s() - case_t0 > hard_limit;
        if (e.empty() && !abort && run >= 0) {
            load.push_back(tl); comp.push_back(tc); total.push_back(tl + tc);
            for (Part& p : parts) { (p.gpu ? gl : cl).push_back(pl[p.gpu]); (p.gpu ? gc : cc).push_back(p.t); }
            if (run == runs - 1 || over) {  // the last run that will complete: digest its result
                TR_CONTEXT(case_id, run, "checksum");
                TR_SCOPE("phase", "\"device\":\"%s\"", dev == CPU ? "cpu" : dev == GPU ? "gpu" : "cpu+gpu");
                try { for (Part& p : parts) r.checksum += p.b->checksum(); have_sum = true; } catch (const std::exception& x) { e = x.what(); }
            }
        }
        if (over && run < runs - 1) abort = true;
        TR_CONTEXT(case_id, run, "unload");
        for (Part& p : parts) p.b->unload();
        if (!e.empty()) { r.err = e; return r; }
    }
    r.aborted = abort;
    r.runs = (int)load.size();
    if (r.runs == 0) { r.err = "stopped at the hard time limit before one run finished"; return r; }
    r.load = Stat::of(load); r.comp = Stat::of(comp); r.total = Stat::of(total);
    r.cpu_load = Stat::of(cl).mean; r.gpu_load = Stat::of(gl).mean;
    r.cpu_comp = Stat::of(cc).mean; r.gpu_comp = Stat::of(gc).mean;
    if (!have_sum) r.checksum = NAN;  // stopped mid-run: there is no finished result to digest
    r.ok = true;
    return r;
}

// ============================================================================
// Devices: core types, affinity
// ============================================================================
static std::vector<int> allowed_cpus() {
    std::vector<int> v;
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &set)) v.push_back(c);
#endif
    if (v.empty()) for (int c = 0; c < omp_get_max_threads(); ++c) v.push_back(c);
    return v;
}

static std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

// Short name of the core a logical CPU is, and a figure to rank core types by.
static std::string core_kind(int cpu, double& rank) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    const std::string midr = read_line(base + "/regs/identification/midr_el1"), cap = read_line(base + "/cpu_capacity");
    rank = cap.empty() ? 0.0 : std::atof(cap.c_str());
    if (!midr.empty()) {
        const unsigned long m = std::strtoul(midr.c_str(), nullptr, 16);
        const unsigned impl = (unsigned)(m >> 24) & 0xff, part = (unsigned)(m >> 4) & 0xfff;
        if (impl == 0x41) {
            switch (part) {
                case 0xd85: return "x925";
                case 0xd87: return "a725";
                case 0xd84: return "x4";
                case 0xd81: return "a720";
                case 0xd80: return "a520";
                default: break;
            }
        }
        char b[32];
        std::snprintf(b, sizeof b, "part%03x", part);
        return b;
    }
    return cap.empty() ? "" : "cap" + cap;
}

static std::string cpu_list(const std::vector<int>& v) {  // "0-4,10-14"
    std::string s;
    for (size_t i = 0; i < v.size();) {
        size_t j = i;
        while (j + 1 < v.size() && v[j + 1] == v[j] + 1) ++j;
        s += (s.empty() ? "" : ",") + std::to_string(v[i]) + (j > i ? "-" + std::to_string(v[j]) : "");
        i = j + 1;
    }
    return s;
}

// Pure devices first (one core type each, then the GPU), mixed ones after.
static std::vector<DevSpec> find_devices(bool have_gpu, int max_threads, const std::string& cpu_sets) {
    const std::vector<int> all = allowed_cpus();
    std::vector<std::string> kinds;
    std::vector<std::vector<int>> members;
    std::vector<double> ranks;
    if (!cpu_sets.empty()) {  // user-defined groups, e.g. "big=0-4,10-14;little=5-9,15-19"
        std::stringstream ss(cpu_sets);
        std::string group;
        while (std::getline(ss, group, ';')) {
            const size_t eq = group.find('=');
            if (eq == std::string::npos || eq == 0) fail("--cpu-sets: expected name=cpulist, got '%s'", group.c_str());
            std::vector<int> cpus;
            std::stringstream ls(group.substr(eq + 1));
            std::string part;
            while (std::getline(ls, part, ',')) {
                const size_t dash = part.find('-');
                const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
                for (int c = a; c <= b; ++c)
                    if (std::find(all.begin(), all.end(), c) != all.end()) cpus.push_back(c);
            }
            if (cpus.empty()) fail("--cpu-sets: group '%s' has no usable CPU", group.c_str());
            kinds.push_back(group.substr(0, eq));
            members.push_back(cpus);
            ranks.push_back(-(double)kinds.size());  // keep the order given
        }
        if (kinds.size() == 1) { kinds.push_back(""); members.emplace_back(); ranks.push_back(-1e9); }  // one group is still a sweep of its own
    }
    for (int c : all) {
        if (!cpu_sets.empty()) break;
        double rank = 0;
        const std::string k = core_kind(c, rank);
        size_t i = 0;
        while (i < kinds.size() && kinds[i] != k) ++i;
        if (i == kinds.size()) { kinds.push_back(k); members.emplace_back(); ranks.push_back(rank); }
        members[i].push_back(c);
        ranks[i] = std::max(ranks[i], rank);
    }
    std::vector<DevSpec> d;
    if (kinds.size() > 1) {
        std::vector<size_t> order(kinds.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ranks[a] > ranks[b]; });
        for (size_t i : order) {
            if (members[i].empty()) continue;
            const int n = std::min<int>(max_threads, (int)members[i].size());
            d.push_back({"cpu:" + kinds[i], CPU, members[i], n, std::to_string(n) + (n == 1 ? " thread" : " threads") + " on the " + kinds[i] + " cores (CPUs " + cpu_list(members[i]) + ")"});
        }
    }
    const int n = std::min<int>(max_threads, (int)all.size());
    const std::string cores = kinds.size() > 1 ? "all core types" : "all cores";
    if (have_gpu) d.push_back({"gpu", GPU, all, n, "GPU alone"});
    d.push_back({"cpu", CPU, all, n, std::to_string(n) + (n == 1 ? " thread" : " threads") + " on " + cores + " (CPUs " + cpu_list(all) + ")"});
    if (have_gpu) d.push_back({"cpu+gpu", BOTH, all, n, "GPU and " + std::to_string(n) + " CPU threads together"});
    if (d.size() == 3 && d[0].kind == GPU) std::swap(d[0], d[1]);  // one core type: cpu, gpu, cpu+gpu
    return d;
}

// Confine every thread of the process to `cpus` and size the thread pools.
// Threads created later inherit the mask from their creator.
static void bind_device(const DevSpec& dv) {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : dv.cpus) CPU_SET(c, &set);
    if (DIR* dir = opendir("/proc/self/task")) {
        while (dirent* e = readdir(dir)) {
            const int tid = std::atoi(e->d_name);
            if (tid > 0) sched_setaffinity(tid, sizeof set, &set);
        }
        closedir(dir);
    }
#endif
    cpu_set_threads(dv.threads);
}

// ============================================================================
// Results file
// ============================================================================
static const char* const CSV_HEADER =
    "task,precision,device,level,shape,iters,runs,units,elements,bytes,work,load_s,compute_s,total_s,"
    "load_min_s,load_max_s,load_sd_s,compute_min_s,compute_max_s,compute_sd_s,cpu_load_s,gpu_load_s,"
    "cpu_compute_s,gpu_compute_s,gpu_share,gops_per_s,cpu_threads,impl,checksum,temp_c,unix_time";

struct Done {  // what a finished case contributes to later decisions
    double load = 0, comp = 0, work = 0, bytes = 0, checksum = 0, share = 0, cpu_comp = 0, gpu_comp = 0;
    long long iters = 1;
    int runs = 0;
};

static std::string case_key(const Case& c, const std::string& dev) {
    return std::string(TASK_NAME[c.task]) + "|" + PREC_NAME[c.prec] + "|" + dev + "|" + c.shape_str() + "|" + std::to_string(c.iters);
}

static std::map<std::string, Done> read_done(const std::string& path) {
    std::map<std::string, Done> m;
    std::ifstream f(path);
    std::string line;
    std::vector<std::string> head;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cell;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, ',')) cell.push_back(x);
        if (head.empty()) { head = cell; continue; }
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < head.size() && i < cell.size(); ++i) row[head[i]] = cell[i];
        Done d;
        d.load = std::atof(row["load_s"].c_str()); d.comp = std::atof(row["compute_s"].c_str());
        d.bytes = std::atof(row["bytes"].c_str()); d.checksum = std::atof(row["checksum"].c_str());
        d.share = std::atof(row["gpu_share"].c_str());
        d.cpu_comp = std::atof(row["cpu_compute_s"].c_str()); d.gpu_comp = std::atof(row["gpu_compute_s"].c_str());
        d.iters = std::max(1LL, std::atoll(row["iters"].c_str())); d.runs = std::atoi(row["runs"].c_str());
        d.work = std::atof(row["work"].c_str()) / (double)d.iters;  // the file stores work for all passes
        m[row["task"] + "|" + row["precision"] + "|" + row["device"] + "|" + row["shape"] + "|" + row["iters"]] = d;
    }
    return m;
}

// ============================================================================
// The sweep
// ============================================================================
static int sweep(const Options& o, const std::string& cmdline) {
    const int max_threads = omp_get_max_threads();
    std::string gpu_desc, why;
    const bool have_gpu = gpu_startup(gpu_desc, why);
    if (!have_gpu) std::printf("GPU unavailable (%s): CPU sweeps only.\n", why.c_str());

    // Which devices to sweep, in order.
    std::vector<DevSpec> offered;
    try { offered = find_devices(have_gpu, max_threads, o.cpu_sets); } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    std::vector<DevSpec> devs;
    if (o.devices.empty()) devs = offered;
    else {
        std::stringstream ss(o.devices);
        std::string tok;
        std::vector<std::string> want;
        while (std::getline(ss, tok, ',')) want.push_back(tok == "both" ? "cpu+gpu" : tok);
        for (const std::string& w : want) {
            bool hit = w == "clusters";
            for (const DevSpec& d : offered) hit = hit || d.name == w;
            if (!hit) {
                std::fprintf(stderr, "error: no device '%s' here. Available:", w.c_str());
                for (const DevSpec& d : offered) std::fprintf(stderr, " %s", d.name.c_str());
                std::fprintf(stderr, " (also: both, clusters)\n");
                return 1;
            }
        }
        for (const DevSpec& d : offered)
            for (const std::string& w : want)
                if (d.name == w || (w == "clusters" && d.name.compare(0, 4, "cpu:") == 0)) { devs.push_back(d); break; }
    }
    if (devs.empty()) { std::fprintf(stderr, "error: no device to run on\n"); return 1; }
    bool kind_used[3] = {false, false, false};
    for (const DevSpec& d : devs) kind_used[d.kind] = true;

    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    std::printf("host        %s\n", host);
    std::printf("cpu         %d threads; matmul: %s\n", max_threads, cpu_describe().c_str());
    if (have_gpu) std::printf("gpu         %s\n", gpu_desc.c_str());
    std::printf("memory      %.1f GiB total, %.1f GiB available; budget %.1f GiB per case\n", meminfo_gib("MemTotal:"), meminfo_gib("MemAvailable:"), o.mem_gb);
    std::printf("each case   %d warm-up + %d measured runs, mean reported; a case predicted to exceed %s ends its series\n", o.warmup, o.runs,
                fmt_time(o.case_budget).c_str());
    std::printf("sweeps      one full pass over all workloads per device, in this order:\n");
    for (size_t i = 0; i < devs.size(); ++i) std::printf("            %zu. %-9s %s\n", i + 1, devs[i].name.c_str(), devs[i].what.c_str());
    int ntp = 0;
    for (int t = 0; t < NTASK; ++t) for (int p = 0; p < NPREC; ++p) ntp += o.task[t] && o.prec[p];
    if (!o.quick && o.repeat && !TR_ON)
        std::printf("plan        %zu series (device x workload x precision), each ending near its budget: very roughly %.0f hours in total\n",
                    devs.size() * ntp, devs.size() * ntp * o.case_budget * 0.75 / 3600.0);
    std::printf("\n");

    // Results file: resume when it exists, unless --fresh.
    std::map<std::string, Done> done, shares;
    if (!o.fresh) done = read_done(o.out);
    if (TR_ON) shares = read_done(o.shares);  // the tracer reuses the benchmark's cpu+gpu splits
    std::FILE* csv = std::fopen(o.out.c_str(), done.empty() ? "w" : "a");
    if (!csv) { std::fprintf(stderr, "error: cannot write %s\n", o.out.c_str()); return 1; }
    if (done.empty()) {
        char date[64];
        const std::time_t tt = std::time(nullptr);
        std::strftime(date, sizeof date, "%Y-%m-%d %H:%M", std::localtime(&tt));
        std::fprintf(csv, "# host: %s\n# date: %s\n# cpu_threads: %d\n# cpu_matmul: %s\n# gpu: %s\n# runs_per_case: %d\n# case_budget_s: %.0f\n", host, date,
                     max_threads, cpu_describe().c_str(), have_gpu ? gpu_desc.c_str() : "none", o.runs, o.case_budget);
        for (const DevSpec& d : offered) std::fprintf(csv, "# device %s: %s\n", d.name.c_str(), d.what.c_str());
        std::fprintf(csv, "%s\n", CSV_HEADER);
        std::fflush(csv);
    } else std::printf("resuming    %zu cases already in %s are kept (use --fresh to start over)\n\n", done.size(), o.out.c_str());

#ifdef GB10_TRACE
    {
        std::string gj;
        gpu_topology_json(gj);
        gpu_trace_config(o.block_detail);
        tr::start(o.trace_out, o.sample_ms, o.slice_us, cmdline, gj, cpu_describe());
    }
#else
    (void)cmdline;
#endif

    // Memory a case needs on a kind of device (GiB), staging included.
    auto need = [&](const Case& c, Kind k, long long ug) {
        double b = (double)c.staging_bytes();
        std::unique_ptr<Backend> cb(make_cpu_backend(c)), gb(have_gpu ? make_gpu_backend(c) : nullptr);
        if (k == CPU) b += (double)cb->bytes(c.units);
        if (k == GPU) b += (double)gb->bytes(c.units);
        if (k == BOTH) b += (double)gb->bytes(ug) + (double)cb->bytes(c.units - ug);
        return b / 1073741824.0;
    };
    // Largest size of each workload x precision that fits on every device in the
    // run, decided once so that all sweeps climb the same ladder.
    const double avail0 = meminfo_gib("MemAvailable:"), budget = avail0 > 0 ? std::min(o.mem_gb, 0.88 * avail0) : o.mem_gb;
    int top[NTASK][NPREC];
    for (int t = 0; t < NTASK; ++t)
        for (int p = 0; p < NPREC; ++p) {
            top[t][p] = -1;
            if (!o.task[t] || !o.prec[p]) continue;
            std::vector<Shape> levels = task_levels((TaskId)t, o.quick);
            if ((int)levels.size() > o.max_level + 1) levels.resize(o.max_level + 1);
            for (int li = 0; li < (int)levels.size(); ++li) {
                const Case c = make_case((TaskId)t, (Prec)p, levels[li], 1);
                double worst = 0;
                for (int k = 0; k < 3; ++k)
                    if (kind_used[k]) worst = std::max(worst, need(c, (Kind)k, std::max<long long>(1, c.units / 2)));
                if (worst > budget) {
                    std::printf("%-10s %-5s %-40s needs %.1f GiB, over the %.1f GiB budget: this size and larger are left out\n", TASK_NAME[t],
                                PREC_NAME[p], c.shape_str().c_str(), worst, budget);
                    break;
                }
                top[t][p] = li;
            }
        }

    const double t_start = now_s();
    int warnings = 0, cases_run = 0, cases_started = 0;

    for (size_t di = 0; di < devs.size(); ++di) {
        const DevSpec& dv = devs[di];
        bind_device(dv);
        std::printf("\n=== sweep %zu of %zu: %s - %s ===\n", di + 1, devs.size(), dv.name.c_str(), dv.what.c_str());
        std::printf("%-10s %-5s %-9s %3s  %-40s %6s %10s %10s %10s %9s  %s\n", "task", "prec", "device", "lvl", "shape", "iters", "load", "compute",
                    "total", "Gop/s", "notes");
        TR_EVENT("device.begin", tr::now_ns(), 0, "\"device\":\"%s\",\"kind\":\"%s\",\"cpu_threads\":%d,\"cpus\":[%s],\"what\":\"%s\"", dv.name.c_str(),
                 dv.kind == CPU ? "cpu" : dv.kind == GPU ? "gpu" : "cpu+gpu", dv.threads,
                 [&] { std::string s; for (int c : dv.cpus) s += (s.empty() ? "" : ",") + std::to_string(c); return s; }().c_str(), dv.what.c_str());

        for (int t = 0; t < NTASK; ++t) {
            for (int pi = 0; pi < NPREC; ++pi) {
                if (top[t][pi] < 0) continue;
                const TaskId task = (TaskId)t;
                const Prec prec = (Prec)pi;
                const std::vector<Shape> levels = task_levels(task, o.quick);
                bool alive = true, have = false;
                Done last;
                double solo[2] = {0, 0}, factor[2] = {1, 1};  // cpu+gpu: solo rates of its halves, and how sharing changed them

                // One rung of the ladder: a size and a pass count.
                auto rung = [&](int level, const Shape& shape, long long iters) {
                    const Case c = make_case(task, prec, shape, iters);
                    const double work = c.work * (double)iters, bytes = (double)c.staging_bytes();
                    const std::string key = case_key(c, dv.name);
                    auto say = [&](const char* fmt, ...) {
                        char b[400];
                        va_list ap;
                        va_start(ap, fmt);
                        std::vsnprintf(b, sizeof b, fmt, ap);
                        va_end(ap);
                        std::printf("%-10s %-5s %-9s %3d  %-40s %6lld  %s\n", TASK_NAME[t], PREC_NAME[pi], dv.name.c_str(), level, c.shape_str().c_str(), iters, b);
                    };
                    Done dn;
                    std::string note;
                    double share = 0.5;  // cpu+gpu: fraction of the units given to the GPU
                    if (dv.kind == BOTH) {
                        // Split by what the cpu and gpu sweeps measured for this very case, corrected
                        // by how the two halves behaved together so far. The tracer prefers the
                        // split the benchmark run actually used, when its CSV is at hand.
                        const auto ic = done.find(case_key(c, "cpu")), ig = done.find(case_key(c, "gpu"));
                        if (ic != done.end()) solo[0] = work / ic->second.comp;
                        if (ig != done.end()) solo[1] = work / ig->second.comp;
                        const auto ir = shares.find(case_key(c, "cpu+gpu"));
                        if (ir != shares.end() && ir->second.share > 0 && ir->second.share < 1) share = ir->second.share;
                        else if (solo[0] > 0 && solo[1] > 0) share = solo[1] * factor[1] / (solo[0] * factor[0] + solo[1] * factor[1]);
                        else note = "even split: no cpu and gpu results for this case";
                    }
                    const bool keep = done.count(key) > 0;
                    if (keep) dn = done[key];
                    else {
                        double per_run = 0;
                        if (have) per_run = last.load * bytes / last.bytes + last.comp * work / (last.work * (double)last.iters);
                        else if (dv.kind == BOTH && solo[0] > 0 && solo[1] > 0) per_run = work / (solo[0] + solo[1]);
                        if (per_run * (o.runs + o.warmup) > o.case_budget) {
                            alive = false;
                            say("series ends: %d runs predicted to take %s (budget %s)", o.runs + o.warmup, fmt_time(per_run * (o.runs + o.warmup)).c_str(),
                                fmt_time(o.case_budget).c_str());
                            return;
                        }
                        long long ug = 0;
                        if (dv.kind == BOTH) {
                            ug = std::min<long long>(std::max<long long>(std::llround(share * (double)c.units), 1), c.units - 1);
                            // Whole groups of four units on each side: cuBLAS's int8 product wants dimensions divisible by 4.
                            if (c.units >= 8 && c.units % 4 == 0) ug = std::min<long long>(std::max<long long>((ug + 2) / 4 * 4, 4), c.units - 4);
                        }
                        const double extra = need(c, dv.kind, ug) - bytes / 1073741824.0, free_now = meminfo_gib("MemAvailable:");
                        if (free_now > 0 && bytes / 1073741824.0 + extra > 0.92 * free_now) {
                            alive = false;
                            say("skipped: needs %.1f GiB, %.1f GiB available", bytes / 1073741824.0 + extra, free_now);
                            return;
                        }
                        Staging st;
                        Result r;
                        TR_CONTEXT(cases_started, 0, "setup");
                        TR_EVENT("case.begin", tr::now_ns(), 0,
                                 "\"task\":\"%s\",\"precision\":\"%s\",\"device\":\"%s\",\"level\":%d,\"shape\":\"%s\",\"iters\":%lld,\"units\":%lld,"
                                 "\"gpu_units\":%lld,\"input_bytes\":%.0f,\"work\":%.6g,\"cpu_threads\":%d",
                                 TASK_NAME[t], PREC_NAME[pi], dv.name.c_str(), level, c.shape_str().c_str(), iters, c.units,
                                 dv.kind == GPU ? c.units : ug, bytes, work, dv.kind == GPU ? 0 : dv.threads);
                        try {
                            generate(c, st);
                            r = run_case(c, dv.kind, st, ug, o.runs, o.warmup, 2.0 * o.case_budget, cases_started);
                        } catch (const std::exception& e) { r.err = e.what(); }
                        release(st);
                        TR_CONTEXT(cases_started, 0, "done");
                        TR_EVENT("case.end", tr::now_ns(), 0, "\"ok\":%s", r.ok ? "true" : "false");
#ifdef GB10_TRACE
                        tr::commit();
#endif
                        TR_CONTEXT(-1, 0, "");
                        ++cases_started;
                        if (!r.ok) {
                            alive = false;
                            ++warnings;
                            say("FAILED: %s (series ends)", r.err.c_str());
                            return;
                        }
                        ++cases_run;
                        dn.load = r.load.mean; dn.comp = r.comp.mean; dn.work = c.work; dn.bytes = bytes; dn.checksum = r.checksum;
                        dn.iters = iters; dn.runs = r.runs; dn.cpu_comp = r.cpu_comp; dn.gpu_comp = r.gpu_comp;
                        dn.share = dv.kind == GPU ? 1.0 : dv.kind == BOTH ? (double)ug / (double)c.units : 0.0;
                        if (r.aborted) { note += std::string(note.empty() ? "" : "; ") + "stopped at the hard time limit after " + std::to_string(r.runs) + " runs"; alive = false; }
                        std::fprintf(csv, "%s,%s,%s,%d,%s,%lld,%d,%lld,%.0f,%.0f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.3e,%.6e,%.6e,%.3e,%.6e,%.6e,%.6e,%.6e,%.4f,%.4f,%d,%s,%.17g,%.1f,%lld\n",
                                     TASK_NAME[t], PREC_NAME[pi], dv.name.c_str(), level, c.shape_str().c_str(), iters, r.runs, c.units, c.elements, bytes, work,
                                     r.load.mean, r.comp.mean, r.total.mean, r.load.mn, r.load.mx, r.load.sd, r.comp.mn, r.comp.mx, r.comp.sd, r.cpu_load,
                                     r.gpu_load, r.cpu_comp, r.gpu_comp, dn.share, work / r.comp.mean / 1e9, dv.kind == GPU ? 0 : dv.threads, r.impl.c_str(),
                                     r.checksum, max_temp_c(), (long long)std::time(nullptr));
                        std::fflush(csv);
                        done[key] = dn;
                    }
                    have = true;
                    last = dn;
                    if (dv.kind == BOTH && dn.cpu_comp > 0 && dn.gpu_comp > 0 && dn.share > 0 && dn.share < 1 && solo[0] > 0 && solo[1] > 0) {
                        // How fast each half really was while sharing the machine, relative to running alone.
                        factor[0] = work * (1.0 - dn.share) / dn.cpu_comp / solo[0];
                        factor[1] = work * dn.share / dn.gpu_comp / solo[1];
                    }
                    if (dv.kind == BOTH) {
                        char b[64];
                        std::snprintf(b, sizeof b, "%sgpu share %.0f%%", note.empty() ? "" : "; ", 100.0 * dn.share);
                        note += b;
                    }
                    // Every device must reach the same result. Integer pipelines agree exactly.
                    const bool exact = prec >= I8 && (task <= T_STENCIL || task == T_LINEAR || task == T_CONV);
                    const double tol = exact ? 1e-12 : prec == F16 ? 1e-2 : 2e-3;
                    for (const DevSpec& other : offered) {
                        const auto it = done.find(case_key(c, other.name));
                        if (other.name == dv.name || it == done.end() || std::isnan(it->second.checksum) || std::isnan(dn.checksum)) continue;
                        if (std::fabs(dn.checksum - it->second.checksum) > tol * std::max(std::fabs(it->second.checksum), 1e-30)) {
                            note += std::string(note.empty() ? "" : "; ") + "WARNING: result differs from " + other.name;
                            ++warnings;
                        }
                        break;
                    }
                    if (keep) note += std::string(note.empty() ? "" : "; ") + "from earlier run";
                    char b[200];
                    std::snprintf(b, sizeof b, "%10s %10s %10s %9.2f  %s", fmt_time(dn.load).c_str(), fmt_time(dn.comp).c_str(), fmt_time(dn.load + dn.comp).c_str(),
                                  work / dn.comp / 1e9, note.c_str());
                    std::printf("%-10s %-5s %-9s %3d  %-40s %6lld %s\n", TASK_NAME[t], PREC_NAME[pi], dv.name.c_str(), level, c.shape_str().c_str(), iters, b);
                    std::fflush(stdout);
                };

                const int tp = top[t][pi];
                if (o.only_level >= 0) {  // the tracer runs exactly one size
                    const int li = std::min(o.only_level, tp);
                    rung(li, levels[li], 1);
                    continue;
                }
                for (int li = 0; li <= tp && alive; ++li) rung(li, levels[li], 1);
                if (o.repeat)
                    for (long long iters = 4, level = tp + 1; alive && iters <= (1LL << 40); iters *= 4, ++level) rung((int)level, levels[tp], iters);
            }
        }
    }
    std::fclose(csv);
#ifdef GB10_TRACE
    tr::stop();
    std::printf("\nTrace: %s\n", o.trace_out.c_str());
#endif
    std::printf("\n%d cases run in %s. Results: %s\n", cases_run, fmt_time(now_s() - t_start).c_str(), o.out.c_str());
    if (warnings) std::printf("%d warning(s) or failure(s): see the notes column.\n", warnings);
    if (o.plot) {
        std::string prefix = o.out;
        const size_t dot = prefix.rfind('.');
        if (dot != std::string::npos && dot > prefix.find_last_of('/') + 1) prefix.erase(dot);
        return plot_stress(o.out, prefix);
    }
    return 0;
}

// ============================================================================
// Command line
// ============================================================================
static void usage(const char* prog) {
    std::printf(
        "usage:\n"
        "  %s [options]                  run the suite\n"
        "  %s plot <csv> [prefix]        redraw the plots from an existing CSV\n\n"
        "what to run:\n"
        "  --tasks LIST        any of sum,histogram,stencil,softmax,linear,conv2d,attention,flash_v1,flash_v2\n"
        "  --precisions LIST   any of fp32,fp16,int8,int4\n"
        "  --devices LIST      any of the device names printed at start-up: cpu, gpu, both (= cpu+gpu), and on a\n"
        "                      CPU with several core types one per type, e.g. cpu:x925; 'clusters' selects those\n"
        "  --cpu-sets SPEC     define the per-core-type CPU devices yourself instead of detecting them,\n"
        "                      e.g. \"big=0-4,10-14;little=5-9,15-19\" gives cpu:big and cpu:little\n"
        "  --max-level N       stop the size ladder at level N (0 is the smallest)\n"
        "  --no-repeat         do not continue past the largest size with more passes\n"
        "how to measure:\n"
        "  --runs N            measured runs per case, the mean is reported (default 25)\n"
        "  --no-warmup         do not discard one untimed run before the measured ones\n"
        "  --case-budget S     seconds allowed for all runs of one case (default 1800)\n"
        "  --mem-gb G          memory budget per case, in GiB (default 90)\n"
        "  --quick             two smallest sizes, one pass, 3 runs: a check that everything works\n"
        "output:\n"
        "  --out FILE          CSV path (default stress.csv); an existing file is resumed\n"
        "  --fresh             overwrite the CSV instead of resuming\n"
        "  --no-plot           skip the SVG plots\n"
#ifdef GB10_TRACE
        "tracer:\n"
        "  --level N           size level to trace (default 1)\n"
        "  --trace-out FILE    event log, JSON Lines (default trace.jsonl)\n"
        "  --shares FILE       benchmark CSV to take the cpu+gpu splits from (default stress.csv)\n"
        "  --sample-ms N       system sampling period, 0 to switch it off (default 20)\n"
        "  --slice-us N        longest CPU work span before it is cut (default 1000)\n"
        "  --block-detail N    kernels with up to N thread blocks also log every block (default 512)\n"
#endif
        "\nCPU threads follow OMP_NUM_THREADS; the CPUs used follow the affinity the program is started with.\n",
        prog, prog);
}

static bool pick(const std::string& list, const char* const* names, int n, bool* out) {
    for (int i = 0; i < n; ++i) out[i] = false;
    std::stringstream ss(list);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        bool hit = false;
        for (int i = 0; i < n; ++i) if (tok == names[i]) out[i] = hit = true;
        if (!hit) { std::fprintf(stderr, "error: unknown name '%s'\n", tok.c_str()); return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<std::string> a(argv + 1, argv + argc);
    if (!a.empty() && a[0] == "plot") {
        if (a.size() < 2) { usage(argv[0]); return 1; }
        std::string prefix = a.size() > 2 ? a[2] : a[1];
        if (a.size() <= 2) {
            const size_t dot = prefix.rfind('.');
            if (dot != std::string::npos && dot > prefix.find_last_of('/') + 1) prefix.erase(dot);
        }
        return plot_stress(a[1], prefix);
    }
    Options o;
    for (bool& b : o.task) b = true;
    for (bool& b : o.prec) b = true;
#ifdef GB10_TRACE
    o.runs = 1; o.warmup = 0; o.only_level = 1; o.plot = false; o.out = "trace_cases.csv"; o.fresh = true;
#endif
    std::string cmdline;
    for (int i = 0; i < argc; ++i) cmdline += (i ? " " : "") + std::string(argv[i]);
    for (size_t i = 0; i < a.size(); ++i) {
        const std::string& k = a[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= a.size()) { std::fprintf(stderr, "error: %s needs a value\n", k.c_str()); std::exit(1); }
            return a[++i];
        };
        if (k == "--tasks") { if (!pick(val(), TASK_NAME, NTASK, o.task)) return 1; }
        else if (k == "--precisions") { if (!pick(val(), PREC_NAME, NPREC, o.prec)) return 1; }
        else if (k == "--devices") o.devices = val();
        else if (k == "--cpu-sets") o.cpu_sets = val();
        else if (k == "--runs") o.runs = std::atoi(val().c_str());
        else if (k == "--no-warmup") o.warmup = 0;
        else if (k == "--case-budget") o.case_budget = std::atof(val().c_str());
        else if (k == "--mem-gb") o.mem_gb = std::atof(val().c_str());
        else if (k == "--max-level") o.max_level = std::atoi(val().c_str());
        else if (k == "--no-repeat") o.repeat = false;
        else if (k == "--out") o.out = val();
        else if (k == "--fresh") o.fresh = true;
        else if (k == "--no-plot") o.plot = false;
        else if (k == "--quick") { o.quick = true; o.runs = 3; o.case_budget = 20; o.repeat = false; }
        else if (k == "--level") o.only_level = std::atoi(val().c_str());
        else if (k == "--trace-out") o.trace_out = val();
        else if (k == "--shares") o.shares = val();
        else if (k == "--sample-ms") o.sample_ms = std::atoi(val().c_str());
        else if (k == "--slice-us") o.slice_us = std::atoi(val().c_str());
        else if (k == "--block-detail") o.block_detail = std::atol(val().c_str());
        else if (k == "-h" || k == "--help") { usage(argv[0]); return 0; }
        else { usage(argv[0]); std::fprintf(stderr, "error: unknown option %s\n", k.c_str()); return 1; }
    }
    if (o.runs < 1 || o.case_budget <= 0 || o.mem_gb <= 0 || o.max_level < 0) { std::fprintf(stderr, "error: bad option value\n"); return 1; }
    return sweep(o, cmdline);
}
