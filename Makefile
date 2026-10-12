# gb10-bench
#
#   make                the two benchmarks: gb10_sweep and gb10_stress
#   make trace          gb10_trace, the instrumented build of the stress suite
#   make trace CUPTI=1  ... with CUPTI records for every CUDA kernel and copy
#   make emu            gb10_stress_emu and gb10_trace_emu: the GPU code path run
#                       on the CPU, for checking the logic on a machine with no GPU
#   make clean
#
# With nvcc on the PATH the binaries cover CPU, GPU and CPU+GPU; without it they
# are CPU-only (and can still draw plots from any CSV).
#
#   make BLAS_LIB=        use the built-in CPU matmul kernels instead of OpenBLAS
#   make NVCC=            force a CPU-only build
#   make TUNE=...         host compiler tuning flags (see below)
#
# After changing any of these, rebuild with `make -B`.

CXX  ?= g++
NVCC ?= nvcc

HAVE_NVCC := $(and $(NVCC),$(shell command -v $(NVCC) 2>/dev/null))
ifneq ($(HAVE_NVCC),)
  rpath = -Xlinker -rpath,$(1)
  CUDA_HOME ?= $(abspath $(dir $(realpath $(HAVE_NVCC)))/..)
else
  rpath = -Wl,-rpath,$(1)
endif

# x86 needs -march=native to reach AVX2/FMA. On aarch64 the baseline already has
# NEON and FMA; -mcpu=native on a core GCC does not know (the GB10's X925/A725)
# can switch the vectoriser to slower SVE code. The stress suite only adds the
# int8 dot-product instructions, which every GB10 core has.
ARCH := $(shell uname -m)
ifeq ($(ARCH),x86_64)
  TUNE ?= -march=native
  STRESS_TUNE ?= $(TUNE)
endif
ifeq ($(ARCH),aarch64)
  STRESS_TUNE ?= -march=armv8.2-a+dotprod
endif

# CPU matmul: OpenBLAS when its development library is installed, otherwise the
# built-in kernels. The OpenMP build of OpenBLAS is preferred because it shares
# one thread pool with the rest of the program.
OMP_BLAS_DIR := /usr/lib/$(shell $(CXX) -print-multiarch 2>/dev/null)/openblas-openmp
ifneq ($(wildcard $(OMP_BLAS_DIR)/libopenblas.so),)
  BLAS_LIB ?= -L$(OMP_BLAS_DIR) $(call rpath,$(OMP_BLAS_DIR)) -lopenblas
  BLAS_LIB_CXX := -L$(OMP_BLAS_DIR) -Wl,-rpath,$(OMP_BLAS_DIR) -lopenblas
else
  BLAS_LIB ?= $(shell echo 'int main(){return 0;}' | $(CXX) -x c++ - -lopenblas -o /dev/null 2>/dev/null && echo -lopenblas)
  BLAS_LIB_CXX := $(BLAS_LIB)
endif
ifneq ($(strip $(BLAS_LIB)),)
  DEFS += -DUSE_BLAS
else
  BLAS_LIB_CXX :=
endif

comma := ,
empty :=
space := $(empty) $(empty)
HOSTFLAGS := $(strip -O3 $(TUNE) -fopenmp)
SFLAGS    := -std=c++17 -O3 $(STRESS_TUNE) -fopenmp $(DEFS)
S         := stress
HDRS      := $(S)/common.hpp $(S)/trace.hpp Makefile

all: gb10_sweep gb10_stress
trace: gb10_trace
emu: gb10_stress_emu gb10_trace_emu

# ---- the original matmul / sum sweep -----------------------------------------
gb10_sweep: gb10_sweep.cpp Makefile
ifneq ($(HAVE_NVCC),)
	$(NVCC) -x cu -std=c++17 -O3 $(DEFS) -Xcompiler $(subst $(space),$(comma),$(HOSTFLAGS)) $< -o $@ \
	    $(call rpath,$(CUDA_HOME)/lib64) -lcublas -lgomp $(BLAS_LIB)
else
	@echo "nvcc not found: building CPU-only"
	$(CXX) -std=c++17 $(HOSTFLAGS) $(DEFS) $< -o $@ $(BLAS_LIB)
endif

# ---- the stress suite ------------------------------------------------------------
# Host code is compiled by $(CXX); only gpu.cu goes through nvcc.
build/%.o: $(S)/%.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(SFLAGS) -c $< -o $@
build/gpu.o: $(S)/gpu.cu $(HDRS)
	@mkdir -p build
	$(NVCC) -x cu -std=c++17 -O3 -c $< -o $@

ifneq ($(HAVE_NVCC),)
gb10_stress: build/main.o build/cpu.o build/plot.o build/gpu.o
	$(NVCC) $^ -o $@ $(call rpath,$(CUDA_HOME)/lib64) -lcublas -lgomp $(BLAS_LIB)
else
gb10_stress: build/main.o build/cpu.o build/plot.o build/nogpu.o
	@echo "nvcc not found: building CPU-only"
	$(CXX) -fopenmp $^ -o $@ $(BLAS_LIB_CXX)
endif

# ---- the tracer: the same sources with every hook switched on ----------------------
ifeq ($(CUPTI),1)
  CUPTI_DIR ?= $(CUDA_HOME)/extras/CUPTI
  TRACE_DEFS := -DHAVE_CUPTI -I$(CUPTI_DIR)/include -I$(CUDA_HOME)/include
  TRACE_LIBS := -L$(CUPTI_DIR)/lib64 $(call rpath,$(CUPTI_DIR)/lib64) -lcupti
endif
build/trace/%.o: $(S)/%.cpp $(HDRS)
	@mkdir -p build/trace
	$(CXX) $(SFLAGS) -DGB10_TRACE $(TRACE_DEFS) -c $< -o $@
build/trace/gpu.o: $(S)/gpu.cu $(HDRS)
	@mkdir -p build/trace
	$(NVCC) -x cu -std=c++17 -O3 -DGB10_TRACE $(TRACE_SM) -c $< -o $@

ifneq ($(HAVE_NVCC),)
gb10_trace: build/trace/main.o build/trace/cpu.o build/trace/plot.o build/trace/trace.o build/trace/gpu.o
	$(NVCC) $^ -o $@ $(call rpath,$(CUDA_HOME)/lib64) -lcublas -lgomp -ldl $(TRACE_LIBS) $(BLAS_LIB)
else
gb10_trace: build/trace/main.o build/trace/cpu.o build/trace/plot.o build/trace/trace.o build/trace/nogpu.o
	@echo "nvcc not found: building CPU-only"
	$(CXX) -fopenmp $^ -o $@ -ldl $(BLAS_LIB_CXX)
endif

# ---- GPU emulation builds (no nvcc involved) ----------------------------------------
build/emu/gpu.o: $(S)/gpu.cu $(S)/cuda_emu.hpp $(HDRS)
	@mkdir -p build/emu
	$(CXX) $(SFLAGS) -DGPU_EMULATE -x c++ -c $< -o $@
build/emu/gpu_trace.o: $(S)/gpu.cu $(S)/cuda_emu.hpp $(HDRS)
	@mkdir -p build/emu
	$(CXX) $(SFLAGS) -DGPU_EMULATE -DGB10_TRACE -x c++ -c $< -o $@
gb10_stress_emu: build/main.o build/cpu.o build/plot.o build/emu/gpu.o
	$(CXX) -fopenmp $^ -o $@ $(BLAS_LIB_CXX)
gb10_trace_emu: build/trace/main.o build/trace/cpu.o build/trace/plot.o build/trace/trace.o build/emu/gpu_trace.o
	$(CXX) -fopenmp $^ -o $@ -ldl $(BLAS_LIB_CXX)

clean:
	rm -rf build gb10_sweep gb10_stress gb10_trace gb10_stress_emu gb10_trace_emu

.PHONY: all trace emu clean
