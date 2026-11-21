# slabitz host C library. Kernels are built separately by tools/build_all.py
# (Slang -> .cu -> .cubin, plus src/sbz_kernels_gen.h from slangc reflection).
#
#   make                -> build/libslabitz.so  (host walk + CUDA backend + stage chain)
#   make build/fuzz_host_walk -> ASan/UBSan fuzz binary for the host walk (tests)
#   make clean         -> remove build artifacts
#
# CFLAGS may come from the environment; WARN and the visibility flag are always
# applied. Public entry points are marked SBZ_API (src/sbz_backend.h).

CC      ?= cc
CFLAGS  ?= -O2
WARN    := -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L
XFLAGS  := $(WARN) -fPIC -fvisibility=hidden
BUILD   := build

HOST_OBJ := $(BUILD)/host_walk.o
LIB_OBJ  := $(HOST_OBJ) $(BUILD)/sbz_cuda.o $(BUILD)/sbz_chain.o

.PHONY: all clean

all: $(BUILD)/libslabitz.so

$(BUILD):
	mkdir -p $(BUILD)

# The walk is internal to the library (hidden like the rest); the tests reach it
# through sbz_dctx_* and the fuzz binary below compiles host_walk.c itself.
$(HOST_OBJ): src/host_walk.c src/host_walk.h | $(BUILD)
	$(CC) $(CFLAGS) $(XFLAGS) -c -o $@ $<

# ASan/UBSan fuzz binary for the bounded host frame walk. Built straight from
# host_walk.c so the walk code under test is the same object the library uses.
$(BUILD)/fuzz_host_walk: tests/fuzz_host_walk.c src/host_walk.c src/host_walk.h | $(BUILD)
	$(CC) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
	    -Isrc -o $@ tests/fuzz_host_walk.c src/host_walk.c

$(BUILD)/sbz_cuda.o: src/sbz_cuda.c src/sbz_backend.h src/sbz_kernels.h src/sbz_kernels_gen.h | $(BUILD)
	$(CC) $(CFLAGS) $(XFLAGS) -c -o $@ $<

$(BUILD)/sbz_chain.o: src/sbz_chain.c src/sbz_chain.h src/sbz_backend.h src/sbz_kernels.h src/sbz_kernels_gen.h src/host_walk.h | $(BUILD)
	$(CC) $(CFLAGS) $(XFLAGS) -c -o $@ $<

# No RPATH in the libraries: they need only libc and libdl. A conda compiler
# (pixi, or a module env) bakes `-rpath <its env>/lib` into its specs file, and
# dlopen("libcuda.so.1") then searches that directory first: from a cvmfs env on a
# machine where that path is broken, every load hammers cvmfs. patchelf strips it;
# the check fails the build if one is left. PATCHELF=true skips it (system gcc).
PATCHELF ?= patchelf
define no_rpath
	$(PATCHELF) --remove-rpath $(1)
	@if readelf -d $(1) | grep -q -E 'RPATH|RUNPATH'; then \
	    echo "error: $(1) still has an RPATH/RUNPATH" >&2; rm -f $(1); exit 1; fi
endef

$(BUILD)/libslabitz.so: $(LIB_OBJ)
	$(CC) -shared -o $@ $^ -ldl
	$(call no_rpath,$@)

clean:
	rm -f $(BUILD)/*.o $(BUILD)/libslabitz.so $(BUILD)/fuzz_host_walk
