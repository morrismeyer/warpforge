# UCX/UCC Performance Benchmark Makefile
#
# Builds highly-tuned UCX and UCC benchmarks to establish the baseline
# performance that warpforge-io should aspire to match.
#
# Prerequisites:
#   - UCX libraries installed (libucp, libucs, libuct)
#   - UCC libraries installed (libucc) - optional, for collective benchmark
#   - GCC with optimization support
#
# Usage:
#   make              # Build with default UCX/UCC location
#   make UCX_HOME=/opt/ucx   # Specify UCX installation
#   make UCC_HOME=/opt/ucc   # Specify UCC installation
#   make ucx          # Build only UCX benchmark
#   make ucc          # Build only UCC benchmark
#   make clean        # Clean build artifacts

# Installation locations (override with UCX_HOME=/path, UCC_HOME=/path)
UCX_HOME ?= /usr/local
UCC_HOME ?= /usr/local

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -lm
CFLAGS += -I$(UCX_HOME)/include -I$(UCC_HOME)/include
LDFLAGS = -L$(UCX_HOME)/lib -L$(UCC_HOME)/lib
LDFLAGS += -Wl,-rpath,$(UCX_HOME)/lib -Wl,-rpath,$(UCC_HOME)/lib
UCX_LDLIBS = -lucp -lucs -luct -lpthread
UCC_LDLIBS = -lucc -lucp -lucs -lpthread -lm

# Alternative: detect UCX/UCC via pkg-config if available
PKG_CONFIG := $(shell which pkg-config 2>/dev/null)
ifneq ($(PKG_CONFIG),)
  UCX_CFLAGS := $(shell pkg-config --cflags ucx 2>/dev/null)
  UCX_LDFLAGS := $(shell pkg-config --libs ucx 2>/dev/null)
  ifneq ($(UCX_CFLAGS),)
    CFLAGS += $(UCX_CFLAGS)
    UCX_LDLIBS = $(UCX_LDFLAGS) -lpthread
  endif
  UCC_CFLAGS := $(shell pkg-config --cflags ucc 2>/dev/null)
  UCC_LDFLAGS := $(shell pkg-config --libs ucc 2>/dev/null)
  ifneq ($(UCC_CFLAGS),)
    CFLAGS += $(UCC_CFLAGS)
    UCC_LDLIBS = $(UCC_LDFLAGS) -lpthread -lm
  endif
endif

# Targets
UCX_TARGETS = ucx_max_bandwidth
UCC_TARGETS = ucc_collective_benchmark
TARGETS = $(UCX_TARGETS) $(UCC_TARGETS)

.PHONY: all ucx ucc clean check check-ucx check-ucc

all: $(TARGETS)

ucx: $(UCX_TARGETS)

ucc: $(UCC_TARGETS)

ucx_max_bandwidth: ucx_max_bandwidth.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(UCX_LDLIBS)

ucc_collective_benchmark: ucc_collective_benchmark.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(UCC_LDLIBS)

# Check for UCX/UCC installation
check: check-ucx check-ucc

check-ucx:
	@echo "Checking UCX installation..."
	@if [ -f "$(UCX_HOME)/include/ucp/api/ucp.h" ]; then \
		echo "  UCX headers found at $(UCX_HOME)/include"; \
	else \
		echo "  ERROR: UCX headers not found at $(UCX_HOME)/include"; \
		echo "  Try: make UCX_HOME=/path/to/ucx/install"; \
		exit 1; \
	fi
	@if [ -f "$(UCX_HOME)/lib/libucp.so" ] || [ -f "$(UCX_HOME)/lib/libucp.dylib" ]; then \
		echo "  UCX libraries found at $(UCX_HOME)/lib"; \
	else \
		echo "  ERROR: UCX libraries not found at $(UCX_HOME)/lib"; \
		exit 1; \
	fi
	@echo "  UCX installation OK"

check-ucc:
	@echo "Checking UCC installation..."
	@if [ -f "$(UCC_HOME)/include/ucc/api/ucc.h" ]; then \
		echo "  UCC headers found at $(UCC_HOME)/include"; \
	else \
		echo "  WARNING: UCC headers not found at $(UCC_HOME)/include"; \
		echo "  UCC benchmark will not be built"; \
		echo "  Try: make UCC_HOME=/path/to/ucc/install"; \
	fi
	@if [ -f "$(UCC_HOME)/lib/libucc.so" ] || [ -f "$(UCC_HOME)/lib/libucc.dylib" ]; then \
		echo "  UCC libraries found at $(UCC_HOME)/lib"; \
	else \
		echo "  WARNING: UCC libraries not found at $(UCC_HOME)/lib"; \
	fi

clean:
	rm -f $(TARGETS) *.o

# Installation helper
install: $(TARGETS)
	install -d $(DESTDIR)/bin
	install -m 755 $(TARGETS) $(DESTDIR)/bin/

# Help
help:
	@echo "UCX/UCC Performance Benchmarks"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all benchmarks (default)"
	@echo "  ucx      - Build UCX bandwidth benchmark only"
	@echo "  ucc      - Build UCC collective benchmark only"
	@echo "  check    - Verify UCX and UCC installation"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install to DESTDIR/bin"
	@echo ""
	@echo "Variables:"
	@echo "  UCX_HOME - UCX installation prefix (default: /usr/local)"
	@echo "  UCC_HOME - UCC installation prefix (default: /usr/local)"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make UCX_HOME=/opt/ucx UCC_HOME=/opt/ucc"
	@echo "  make ucx UCX_HOME=\$HOME/ucx/install"
	@echo "  make ucc UCC_HOME=\$HOME/ucc/install"
	@echo ""
	@echo "Benchmarks:"
	@echo "  ucx_max_bandwidth       - Point-to-point UCX bandwidth test"
	@echo "  ucc_collective_benchmark - UCC collective operations test"
