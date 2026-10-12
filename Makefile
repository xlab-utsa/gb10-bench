# Builds gb10_sweep. With nvcc on the PATH you get CPU + GPU + CPU+GPU;
# without it, a CPU-only binary (which can still draw plots from any CSV).
#
#   make                  auto-detect nvcc and OpenBLAS
#   make BLAS_LIB=        force the built-in CPU matmul kernel
#   make NVCC=            force a CPU-only build
#   make TUNE=-mcpu=...   pass a CPU tuning flag to the host compiler
#
# After changing any of these, rebuild with `make -B`.

CXX  ?= g++
NVCC ?= nvcc

HAVE_NVCC := $(and $(NVCC),$(shell command -v $(NVCC) 2>/dev/null))
ifneq ($(HAVE_NVCC),)
  rpath = -Xlinker -rpath,$(1)
else
  rpath = -Wl,-rpath,$(1)
endif

# x86 needs -march=native to reach AVX2/FMA. On aarch64 the baseline already
# has NEON and FMA, and -mcpu=native on a core GCC does not know (the GB10's
# X925/A725) can switch the vectoriser to slower SVE code, so no flag there.
ifeq ($(shell uname -m),x86_64)
  TUNE ?= -march=native
endif

# CPU matmul: OpenBLAS when its development library is installed, otherwise the
# built-in kernel. The OpenMP build of OpenBLAS is preferred because it shares
# one thread pool with the rest of the program.
OMP_BLAS_DIR := /usr/lib/$(shell $(CXX) -print-multiarch 2>/dev/null)/openblas-openmp
ifneq ($(wildcard $(OMP_BLAS_DIR)/libopenblas.so),)
  BLAS_LIB ?= -L$(OMP_BLAS_DIR) $(call rpath,$(OMP_BLAS_DIR)) -lopenblas
else
  BLAS_LIB ?= $(shell echo 'int main(){return 0;}' | $(CXX) -x c++ - -lopenblas -o /dev/null 2>/dev/null && echo -lopenblas)
endif
ifneq ($(strip $(BLAS_LIB)),)
  DEFS += -DUSE_BLAS
endif

comma := ,
empty :=
space := $(empty) $(empty)
HOSTFLAGS := $(strip -O3 $(TUNE) -fopenmp)

gb10_sweep: gb10_sweep.cpp Makefile
ifneq ($(HAVE_NVCC),)
	$(NVCC) -x cu -std=c++17 -O3 $(DEFS) -Xcompiler $(subst $(space),$(comma),$(HOSTFLAGS)) $< -o $@ \
	    $(call rpath,$(abspath $(dir $(realpath $(HAVE_NVCC)))/../lib64)) -lcublas -lgomp $(BLAS_LIB)
else
	@echo "nvcc not found: building CPU-only"
	$(CXX) -std=c++17 $(HOSTFLAGS) $(DEFS) $< -o $@ $(BLAS_LIB)
endif

clean:
	rm -f gb10_sweep

.PHONY: clean
