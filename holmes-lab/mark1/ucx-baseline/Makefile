# UCX Maximum Bandwidth Benchmark Makefile
#
# Builds a highly-tuned UCX bandwidth test to establish the baseline
# performance that warpforge-io should aspire to match.
#
# Prerequisites:
#   - UCX libraries installed (libucp, libucs, libuct)
#   - GCC with optimization support
#
# Usage:
#   make              # Build with default UCX location
#   make UCX_HOME=/opt/ucx   # Specify UCX installation
#   make clean        # Clean build artifacts

# UCX installation location (override with UCX_HOME=/path)
UCX_HOME ?= /usr/local

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
CFLAGS += -I$(UCX_HOME)/include
LDFLAGS = -L$(UCX_HOME)/lib -Wl,-rpath,$(UCX_HOME)/lib
LDLIBS = -lucp -lucs -luct -lpthread

# Alternative: detect UCX via pkg-config if available
PKG_CONFIG := $(shell which pkg-config 2>/dev/null)
ifneq ($(PKG_CONFIG),)
  UCX_CFLAGS := $(shell pkg-config --cflags ucx 2>/dev/null)
  UCX_LDFLAGS := $(shell pkg-config --libs ucx 2>/dev/null)
  ifneq ($(UCX_CFLAGS),)
    CFLAGS += $(UCX_CFLAGS)
    LDFLAGS =
    LDLIBS = $(UCX_LDFLAGS) -lpthread
  endif
endif

# Targets
TARGETS = ucx_max_bandwidth

.PHONY: all clean check

all: $(TARGETS)

ucx_max_bandwidth: ucx_max_bandwidth.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

# Check for UCX headers
check:
	@echo "Checking UCX installation..."
	@if [ -f "$(UCX_HOME)/include/ucp/api/ucp.h" ]; then \
		echo "UCX headers found at $(UCX_HOME)/include"; \
	else \
		echo "ERROR: UCX headers not found at $(UCX_HOME)/include"; \
		echo "Try: make UCX_HOME=/path/to/ucx/install"; \
		exit 1; \
	fi
	@if [ -f "$(UCX_HOME)/lib/libucp.so" ]; then \
		echo "UCX libraries found at $(UCX_HOME)/lib"; \
	else \
		echo "ERROR: UCX libraries not found at $(UCX_HOME)/lib"; \
		exit 1; \
	fi
	@echo "UCX installation OK"

clean:
	rm -f $(TARGETS) *.o

# Installation helper
install: $(TARGETS)
	install -d $(DESTDIR)/bin
	install -m 755 $(TARGETS) $(DESTDIR)/bin/

# Help
help:
	@echo "UCX Maximum Bandwidth Benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build ucx_max_bandwidth (default)"
	@echo "  check    - Verify UCX installation"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install to DESTDIR/bin"
	@echo ""
	@echo "Variables:"
	@echo "  UCX_HOME - UCX installation prefix (default: /usr/local)"
	@echo ""
	@echo "Examples:"
	@echo "  make"
	@echo "  make UCX_HOME=/opt/ucx"
	@echo "  make check UCX_HOME=\$HOME/ucx/install"
