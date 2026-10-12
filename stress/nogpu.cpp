// nogpu.cpp - stands in for gpu.cu in a CPU-only build.
#include "common.hpp"
Backend* make_gpu_backend(const Case&) { return nullptr; }
bool gpu_startup(std::string&, std::string& why) { why = "built without nvcc"; return false; }
void gpu_topology_json(std::string& out) { out.clear(); }
void gpu_trace_config(long) {}
void gpu_thread_init() {}
