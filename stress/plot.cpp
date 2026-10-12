// plot.cpp - draws the gb10_stress results as SVG. It only reads the CSV and is
// never active while anything is being timed.
//
// Four files: <prefix>_total.svg, _compute.svg, _load.svg (seconds per run) and
// _rate.svg (operations per second of compute). Each is a grid with one row per
// workload and one column per precision; the panels of a row share both axes.
// Open them in a browser and hover a point for its numbers.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    std::string task, prec, dev, shape;
    double iters = 1, runs = 0, bytes = 0, work = 0, load = 0, comp = 0, total = 0, load_sd = 0, comp_sd = 0, share = 0, rate = 0;
};
struct Point { double x, y; std::string tip; };
struct Curve { int dev; std::vector<Point> pts; };

const char* const TASKS[] = {"sum", "histogram", "stencil", "softmax", "linear", "conv2d", "attention", "flash_v1", "flash_v2"};
const char* const PRECS[] = {"fp32", "fp16", "int8", "int4"};
// Colour slots in a fixed order: 0 cpu (all cores), 1 gpu, 2 cpu+gpu, then one per
// CPU core type. Each slot also has its own marker shape, so a series can be
// told apart without relying on colour.
enum { NSLOT = 8 };
const char* const LIGHT[NSLOT] = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"};
const char* const DARK[NSLOT] = {"#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"};

struct Dev { std::string name, label; int slot; };

// Devices in the order the sweeps run: single core types, GPU, all cores, CPU+GPU.
std::vector<Dev> devices_of(const std::vector<Row>& rows) {
    std::vector<std::string> types;
    bool cpu = false, gpu = false, both = false;
    for (const Row& r : rows) {
        if (r.dev == "cpu") cpu = true;
        else if (r.dev == "gpu") gpu = true;
        else if (r.dev == "cpu+gpu") both = true;
        else if (std::find(types.begin(), types.end(), r.dev) == types.end()) types.push_back(r.dev);
    }
    std::vector<Dev> d;
    for (size_t i = 0; i < types.size(); ++i) {
        std::string core = types[i].compare(0, 4, "cpu:") == 0 ? types[i].substr(4) : types[i];
        for (char& ch : core) ch = (char)std::toupper((unsigned char)ch);
        d.push_back({types[i], "CPU, " + core + " cores only", (int)std::min<size_t>(3 + i, NSLOT - 1)});
    }
    if (gpu) d.push_back({"gpu", "GPU", 1});
    if (cpu) d.push_back({"cpu", types.empty() ? "CPU" : "CPU, all cores", 0});
    if (both) d.push_back({"cpu+gpu", "CPU+GPU", 2});
    return d;
}

// A data marker: filled in the series colour, with a ring in the surface colour.
void marker(std::ostringstream& s, int slot, double x, double y) {
    char b[400];
    const char* common = "stroke=\"#fcfcfb\" stroke-width=\"1.5\" pointer-events=\"none\"";
    switch (slot % 5) {
        case 0: std::snprintf(b, sizeof b, "<circle class=\"mk f%d ring\" cx=\"%.1f\" cy=\"%.1f\" r=\"4\" fill=\"%s\" %s/>", slot, x, y, LIGHT[slot], common); break;
        case 1: std::snprintf(b, sizeof b, "<rect class=\"mk f%d ring\" x=\"%.1f\" y=\"%.1f\" width=\"7.4\" height=\"7.4\" fill=\"%s\" %s/>", slot, x - 3.7, y - 3.7, LIGHT[slot], common); break;
        case 2: std::snprintf(b, sizeof b, "<polygon class=\"mk f%d ring\" points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" %s/>", slot, x, y - 5.0, x + 4.8, y + 3.4, x - 4.8, y + 3.4, LIGHT[slot], common); break;
        case 3: std::snprintf(b, sizeof b, "<polygon class=\"mk f%d ring\" points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" %s/>", slot, x, y - 5.2, x + 5.2, y, x, y + 5.2, x - 5.2, y, LIGHT[slot], common); break;
        default: std::snprintf(b, sizeof b, "<polygon class=\"mk f%d ring\" points=\"%.1f,%.1f %.1f,%.1f %.1f,%.1f\" fill=\"%s\" %s/>", slot, x, y + 5.0, x + 4.8, y - 3.4, x - 4.8, y - 3.4, LIGHT[slot], common); break;
    }
    s << b;
}

enum Metric { M_TOTAL, M_COMPUTE, M_LOAD, M_RATE };

std::string esc(const std::string& s) {
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
std::string num(double v, const char* fmt = "%.3g") { char b[48]; std::snprintf(b, sizeof b, fmt, v); return b; }
std::string px(double v) { return num(v, "%.1f"); }

std::string fmt_time(double s) {
    const char* unit = "s";
    double v = s;
    const double e = 1.000001;
    if (s * e < 1e-6) { v = s * 1e9; unit = "ns"; }
    else if (s * e < 1e-3) { v = s * 1e6; unit = "\xC2\xB5s"; }
    else if (s * e < 1.0) { v = s * 1e3; unit = "ms"; }
    return num(v) + " " + unit;
}
std::string fmt_count(double v) {
    if (v >= 0.9995e15) return num(v, "%.0e");
    static const char* suffix[] = {"", "k", "M", "B", "T"};
    int i = 0;
    while (v >= 999.5 && i < 4) { v /= 1000.0; ++i; }
    return num(v) + suffix[i];
}
std::string fmt_bytes(double v) {  // decimal units, so that decade ticks are round
    static const char* suffix[] = {"B", "kB", "MB", "GB", "TB"};
    int i = 0;
    while (v >= 999.5 && i < 4) { v /= 1000.0; ++i; }
    return num(v) + " " + suffix[i];
}

struct Tick { double v; bool labelled; };
std::vector<Tick> log_ticks(double lo, double hi, int max_labels) {
    const int e0 = (int)std::floor(std::log10(lo)), e1 = (int)std::ceil(std::log10(hi));
    int decades = 0;
    for (int e = e0; e <= e1; ++e) { const double v = std::pow(10.0, e); if (v >= lo * 0.999 && v <= hi * 1.001) ++decades; }
    const int every = std::max(1, (decades + max_labels - 1) / max_labels);
    std::vector<Tick> t;
    for (int e = e0; e <= e1; ++e)
        for (int m : {1, 2, 5}) {
            if (m != 1 && decades >= 3) continue;
            const double v = m * std::pow(10.0, e);
            if (v < lo * 0.999 || v > hi * 1.001) continue;
            t.push_back({v, m != 1 || ((e % every) + every) % every == 0});
        }
    return t;
}

bool load(const std::string& path, std::vector<Row>& rows, std::map<std::string, std::string>& meta) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    std::vector<std::string> head;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t colon = line.find(':');
            if (colon != std::string::npos && colon >= 2) meta[line.substr(2, colon - 2)] = line.substr(std::min(line.size(), colon + 2));
            continue;
        }
        std::vector<std::string> cell;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, ',')) cell.push_back(x);
        if (head.empty()) { head = cell; continue; }
        std::map<std::string, std::string> m;
        for (size_t i = 0; i < head.size() && i < cell.size(); ++i) m[head[i]] = cell[i];
        Row r;
        auto d = [&](const char* k) { return std::atof(m[k].c_str()); };
        r.task = m["task"]; r.prec = m["precision"]; r.dev = m["device"]; r.shape = m["shape"];
        r.iters = d("iters"); r.runs = d("runs"); r.bytes = d("bytes"); r.work = d("work");
        r.load = d("load_s"); r.comp = d("compute_s"); r.total = d("total_s");
        r.load_sd = d("load_sd_s"); r.comp_sd = d("compute_sd_s"); r.share = d("gpu_share"); r.rate = d("gops_per_s");
        if (r.work > 0 && r.comp > 0 && r.load > 0) rows.push_back(r);
    }
    return true;
}

int render(const std::vector<Row>& rows, std::map<std::string, std::string>& meta, Metric metric, const std::string& path) {
    static const char* const TITLE[] = {"Total time per run (load + compute)", "Compute time per run", "Memory load time per run",
                                        "Compute throughput"};
    static const char* const YUNIT[] = {"seconds", "seconds", "seconds", "billions of operations per second"};
    static const char* const XLABEL[] = {"operations per run", "operations per run", "input bytes", "operations per run"};

    // ---- collect curves: [task][precision] -> one curve per device --------------
    const std::vector<Dev> devs = devices_of(rows);
    std::vector<int> tasks;
    std::map<std::pair<int, int>, std::vector<Curve>> panels;
    std::map<int, std::pair<double, double>> xr, yr;
    for (int t = 0; t < 9; ++t) {
        bool any = false;
        for (int p = 0; p < 4; ++p) {
            std::vector<Curve> cs;
            for (const Dev& dv : devs) {
                Curve c{dv.slot, {}};
                std::map<double, std::vector<const Row*>> by_bytes;
                for (const Row& r : rows) {
                    if (r.task != TASKS[t] || r.prec != PRECS[p] || r.dev != dv.name) continue;
                    std::string tip = dv.label + " \xC2\xB7 " + r.shape + " \xC2\xB7 " + num(r.iters, "%.0f") + (r.iters == 1 ? " pass" : " passes") +
                                      " \xC2\xB7 mean of " + num(r.runs, "%.0f") + " runs \xC2\xB7 load " + fmt_time(r.load) + " (\xC2\xB1" + fmt_time(r.load_sd) +
                                      ") \xC2\xB7 compute " + fmt_time(r.comp) + " (\xC2\xB1" + fmt_time(r.comp_sd) + ") \xC2\xB7 total " + fmt_time(r.total) +
                                      " \xC2\xB7 " + num(r.rate) + " Gop/s \xC2\xB7 input " + fmt_bytes(r.bytes);
                    if (dv.name == "cpu+gpu") tip += " \xC2\xB7 GPU share " + num(100.0 * r.share, "%.0f") + "%";
                    if (metric == M_LOAD) by_bytes[r.bytes].push_back(&r);
                    else c.pts.push_back({r.work, metric == M_TOTAL ? r.total : metric == M_COMPUTE ? r.comp : r.rate, tip});
                }
                // Load time does not depend on the pass count: one point per data size.
                for (const auto& kv : by_bytes) {
                    double s = 0;
                    for (const Row* r : kv.second) s += r->load;
                    const Row* r = kv.second.front();
                    c.pts.push_back({kv.first, s / kv.second.size(), dv.label + " \xC2\xB7 " + r->shape + " \xC2\xB7 input " +
                                                                      fmt_bytes(r->bytes) + " \xC2\xB7 load " + fmt_time(s / kv.second.size())});
                }
                std::sort(c.pts.begin(), c.pts.end(), [](const Point& a, const Point& b) { return a.x < b.x; });
                if (c.pts.empty()) continue;
                any = true;
                for (const Point& q : c.pts) {
                    if (!xr.count(t)) { xr[t] = {q.x, q.x}; yr[t] = {q.y, q.y}; }
                    xr[t].first = std::min(xr[t].first, q.x); xr[t].second = std::max(xr[t].second, q.x);
                    yr[t].first = std::min(yr[t].first, q.y); yr[t].second = std::max(yr[t].second, q.y);
                }
                cs.push_back(c);
            }
            panels[{t, p}] = cs;
        }
        if (any) tasks.push_back(t);
    }
    if (tasks.empty()) return 0;

    // ---- geometry -------------------------------------------------------------------
    const double pad_l = 28, pad_r = 20, head_h = 134, gutter = 56, plot_w = 290, gap_x = 22, title_h = 26, plot_h = 150, row_h = title_h + plot_h + 44;
    const double col_w = gutter + plot_w + gap_x, W = pad_l + 4 * col_w + pad_r - gap_x, H = head_h + tasks.size() * row_h + 56;

    std::ostringstream s;
    s << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << W << " " << H << "\" width=\"" << W << "\" height=\"" << H
      << "\" font-family=\"system-ui, -apple-system, 'Segoe UI', sans-serif\">\n";
    s << "<style>\n@media (prefers-color-scheme: dark) {\n"
         "  .bg{fill:#1a1a19} .ink{fill:#f0efec} .ink2{fill:#c3c2b7} .grid{stroke:#2c2c2a} .axis{stroke:#383835} .ring{stroke:#1a1a19}\n";
    for (int d = 0; d < NSLOT; ++d) s << "  .l" << d << "{stroke:" << DARK[d] << "} .f" << d << "{fill:" << DARK[d] << "}\n";
    s << "}\n.tick{font-size:11px;font-variant-numeric:tabular-nums}\n"
         ".mk{transform-box:fill-box;transform-origin:center}\n.hit:hover + .mk{transform:scale(1.4)}\n</style>\n";
    s << "<rect class=\"bg\" width=\"" << W << "\" height=\"" << H << "\" fill=\"#fcfcfb\"/>\n";

    std::string sub;
    auto add = [&](const std::string& part) { if (!part.empty()) sub += (sub.empty() ? "" : "  \xC2\xB7  ") + part; };
    if (meta.count("host")) add(meta["host"]);
    if (meta.count("gpu") && meta["gpu"] != "none") add("GPU: " + meta["gpu"]);
    if (meta.count("cpu_threads")) add(meta["cpu_threads"] + " CPU threads");
    if (meta.count("cpu_matmul")) add("CPU matmul: " + meta["cpu_matmul"]);
    if (meta.count("runs_per_case")) add("mean of " + meta["runs_per_case"] + " runs");
    if (meta.count("date")) add(meta["date"]);
    s << "<text class=\"ink\" x=\"" << pad_l << "\" y=\"36\" font-size=\"20\" font-weight=\"600\" fill=\"#0b0b0b\">" << TITLE[metric]
      << "</text>\n<text class=\"ink2\" x=\"" << pad_l << "\" y=\"58\" font-size=\"12.5\" fill=\"#52514e\">" << esc(sub) << "</text>\n";
    double lx = pad_l;
    for (const Dev& dv : devs) {
        const int d = dv.slot;
        s << "<line class=\"l" << d << "\" x1=\"" << lx << "\" x2=\"" << lx + 26 << "\" y1=\"82\" y2=\"82\" stroke=\"" << LIGHT[d]
          << "\" stroke-width=\"2\" stroke-linecap=\"round\"/>";
        marker(s, d, lx + 13, 82);
        s << "<text class=\"ink\" x=\"" << lx + 34 << "\" y=\"86.5\" font-size=\"13\" fill=\"#0b0b0b\">" << esc(dv.label) << "</text>\n";
        lx += 34 + 7.0 * dv.label.size() + 28;
    }
    s << "<text class=\"ink2\" x=\"" << pad_l << "\" y=\"108\" font-size=\"12\" fill=\"#52514e\">y: " << YUNIT[metric]
      << "  \xC2\xB7  x: " << XLABEL[metric] << "  \xC2\xB7  both logarithmic; panels in a row share axes</text>\n";

    // ---- panels ----------------------------------------------------------------------
    for (size_t ri = 0; ri < tasks.size(); ++ri) {
        const int t = tasks[ri];
        double xlo = xr[t].first, xhi = xr[t].second, ylo = yr[t].first, yhi = yr[t].second;
        if (xhi <= xlo) xhi = xlo * 10;
        const double lx0 = std::log10(xlo / 1.8), lx1 = std::log10(xhi * 1.8), ly0 = std::log10(ylo / 1.8), ly1 = std::log10(yhi * 1.8);
        for (int p = 0; p < 4; ++p) {
            const std::vector<Curve>& cs = panels[{t, p}];
            const double x0 = pad_l + p * col_w + gutter, y0 = head_h + ri * row_h + title_h;
            s << "<text class=\"ink\" x=\"" << px(x0 - gutter) << "\" y=\"" << px(y0 - 10) << "\" font-size=\"13\" font-weight=\"600\" fill=\"#0b0b0b\">"
              << TASKS[t] << "</text><text class=\"ink2\" x=\"" << px(x0 - gutter + 8.0 * std::strlen(TASKS[t]) + 8) << "\" y=\"" << px(y0 - 10)
              << "\" font-size=\"12.5\" fill=\"#52514e\">" << PRECS[p] << "</text>\n";
            auto X = [&](double v) { return x0 + (std::log10(v) - lx0) / (lx1 - lx0) * plot_w; };
            auto Y = [&](double v) { return y0 + plot_h - (std::log10(v) - ly0) / (ly1 - ly0) * plot_h; };
            for (const Tick& tk : log_ticks(std::pow(10, ly0), std::pow(10, ly1), 6)) {
                s << "<line class=\"grid\" x1=\"" << px(x0) << "\" x2=\"" << px(x0 + plot_w) << "\" y1=\"" << px(Y(tk.v)) << "\" y2=\"" << px(Y(tk.v))
                  << "\" stroke=\"#e1e0d9\"/>";
                if (tk.labelled && p == 0)
                    s << "<text class=\"tick\" x=\"" << px(x0 - 8) << "\" y=\"" << px(Y(tk.v) + 4) << "\" text-anchor=\"end\" fill=\"#898781\">"
                      << (metric == M_RATE ? fmt_count(tk.v) : fmt_time(tk.v)) << "</text>";
                s << "\n";
            }
            s << "<line class=\"axis\" x1=\"" << px(x0) << "\" x2=\"" << px(x0 + plot_w) << "\" y1=\"" << px(y0 + plot_h) << "\" y2=\"" << px(y0 + plot_h)
              << "\" stroke=\"#c3c2b7\"/>\n";
            for (const Tick& tk : log_ticks(std::pow(10, lx0), std::pow(10, lx1), 6)) {
                if (!tk.labelled) continue;
                s << "<line class=\"axis\" x1=\"" << px(X(tk.v)) << "\" x2=\"" << px(X(tk.v)) << "\" y1=\"" << px(y0 + plot_h) << "\" y2=\""
                  << px(y0 + plot_h + 4) << "\" stroke=\"#c3c2b7\"/><text class=\"tick\" x=\"" << px(X(tk.v)) << "\" y=\"" << px(y0 + plot_h + 18)
                  << "\" text-anchor=\"middle\" fill=\"#898781\">" << (metric == M_LOAD ? fmt_bytes(tk.v) : fmt_count(tk.v)) << "</text>\n";
            }
            if (cs.empty()) {
                s << "<text class=\"ink2\" x=\"" << px(x0 + plot_w / 2) << "\" y=\"" << px(y0 + plot_h / 2) << "\" font-size=\"12\" text-anchor=\"middle\" fill=\"#52514e\">not run</text>\n";
                continue;
            }
            for (const Curve& c : cs) {
                s << "<polyline class=\"l" << c.dev << "\" fill=\"none\" stroke=\"" << LIGHT[c.dev]
                  << "\" stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"";
                for (const Point& q : c.pts) s << px(X(q.x)) << "," << px(Y(q.y)) << " ";
                s << "\"/>\n";
            }
            for (const Curve& c : cs)
                for (const Point& q : c.pts) {
                    s << "<g><title>" << esc(q.tip) << "</title><circle class=\"hit\" cx=\"" << px(X(q.x)) << "\" cy=\"" << px(Y(q.y))
                      << "\" r=\"10\" fill=\"transparent\"/>";
                    marker(s, c.dev, X(q.x), Y(q.y));
                    s << "</g>\n";
                }
        }
    }
    const double fy = head_h + tasks.size() * row_h + 20;
    s << "<text class=\"ink2\" x=\"" << pad_l << "\" y=\"" << fy << "\" font-size=\"11.5\" fill=\"#52514e\">"
         "Each point is one case: the mean over its runs. Load is allocating device memory and copying the operands in; compute is the passes "
         "of the workload. Hover a point for its values and spread.</text>\n</svg>\n";

    std::ofstream out(path);
    if (!out) { std::fprintf(stderr, "error: cannot write %s\n", path.c_str()); return 1; }
    out << s.str();
    std::printf("Plot: %s\n", path.c_str());
    return 0;
}

}  // namespace

int plot_stress(const std::string& csv_path, const std::string& prefix) {
    std::vector<Row> rows;
    std::map<std::string, std::string> meta;
    if (!load(csv_path, rows, meta)) { std::fprintf(stderr, "error: cannot read %s\n", csv_path.c_str()); return 1; }
    if (rows.empty()) { std::fprintf(stderr, "error: no data rows in %s\n", csv_path.c_str()); return 1; }
    int rc = 0;
    rc |= render(rows, meta, M_TOTAL, prefix + "_total.svg");
    rc |= render(rows, meta, M_COMPUTE, prefix + "_compute.svg");
    rc |= render(rows, meta, M_LOAD, prefix + "_load.svg");
    rc |= render(rows, meta, M_RATE, prefix + "_rate.svg");
    return rc;
}
