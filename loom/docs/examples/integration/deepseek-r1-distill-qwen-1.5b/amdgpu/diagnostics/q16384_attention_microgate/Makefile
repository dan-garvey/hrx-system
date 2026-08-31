CC ?= gcc
PYTHON ?= python3

BUILD_DIR := build
BIN_DIR := bin
RUNNER := $(BIN_DIR)/q16384_attention_microgate
INTEGRATION_DIR := $(if $(wildcard src/q16k_aiter_integration.c),src,../..)
CPPFLAGS := -Isrc -I$(INTEGRATION_DIR) -isystem toolchain/include
CFLAGS := -std=c11 -O2 -g -Wall -Wextra -Werror -Wpedantic
LDFLAGS := -Ltoolchain/lib -Wl,--enable-new-dtags,-rpath,/opt/rocm/lib -Wl,-z,now -Wl,--as-needed
LDLIBS := -l:libhsa-runtime64.so.1 -ldl -lm -pthread
RUNNER_SOURCES := \
	src/q16384_attention_microgate.c \
	src/reference.c \
	src/sha256.c \
	$(INTEGRATION_DIR)/q16k_aiter_integration.c
RUNNER_OBJECTS := \
	$(BUILD_DIR)/q16384_attention_microgate.o \
	$(BUILD_DIR)/reference.o \
	$(BUILD_DIR)/sha256.o \
	$(BUILD_DIR)/q16k_aiter_integration.o

.PHONY: all check clean offline-review test-reference test-reference-sanitize test-interlocks test-interlocks-sanitize test-python test-scripts test-self-check verify-inputs verify-manifest

all: $(RUNNER)

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

$(BUILD_DIR)/q16384_attention_microgate.o: src/q16384_attention_microgate.c src/reference.h src/sha256.h $(INTEGRATION_DIR)/q16k_aiter_integration.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/reference.o: src/reference.c src/reference.h | $(BUILD_DIR)
	$(CC) -Isrc $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/sha256.o: src/sha256.c src/sha256.h | $(BUILD_DIR)
	$(CC) -Isrc $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/q16k_aiter_integration.o: $(INTEGRATION_DIR)/q16k_aiter_integration.c $(INTEGRATION_DIR)/q16k_aiter_integration.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(RUNNER): $(RUNNER_OBJECTS) | $(BIN_DIR)
	$(CC) $(RUNNER_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_reference: tests/test_reference.c src/reference.c src/reference.h src/sha256.c src/sha256.h | $(BUILD_DIR)
	$(CC) -Isrc $(CFLAGS) tests/test_reference.c src/reference.c src/sha256.c -lm -o $@

$(BUILD_DIR)/test_reference_sanitize: tests/test_reference.c src/reference.c src/reference.h src/sha256.c src/sha256.h | $(BUILD_DIR)
	$(CC) -Isrc $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_reference.c src/reference.c src/sha256.c -lm -o $@

test-reference: $(BUILD_DIR)/test_reference
	./$(BUILD_DIR)/test_reference

test-reference-sanitize: $(BUILD_DIR)/test_reference_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD_DIR)/test_reference_sanitize

$(BUILD_DIR)/test_interlocks: tests/test_interlocks.c src/q16384_attention_microgate.c src/reference.c src/reference.h src/sha256.c src/sha256.h $(INTEGRATION_DIR)/q16k_aiter_integration.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_interlocks.c src/reference.c src/sha256.c $(INTEGRATION_DIR)/q16k_aiter_integration.c $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_interlocks_sanitize: tests/test_interlocks.c src/q16384_attention_microgate.c src/reference.c src/reference.h src/sha256.c src/sha256.h $(INTEGRATION_DIR)/q16k_aiter_integration.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_interlocks.c src/reference.c src/sha256.c $(INTEGRATION_DIR)/q16k_aiter_integration.c $(LDFLAGS) $(LDLIBS) -o $@

test-interlocks: $(BUILD_DIR)/test_interlocks
	LD_LIBRARY_PATH=$(CURDIR)/toolchain/lib ./$(BUILD_DIR)/test_interlocks

test-interlocks-sanitize: $(BUILD_DIR)/test_interlocks_sanitize
	LD_LIBRARY_PATH=$(CURDIR)/toolchain/lib ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./$(BUILD_DIR)/test_interlocks_sanitize

test-self-check: $(RUNNER)
	LD_LIBRARY_PATH=$(CURDIR)/toolchain/lib ./$(RUNNER) --self-check "$(CURDIR)"

verify-manifest:
	$(PYTHON) -I scripts/package_manifest.py "$(CURDIR)"

verify-inputs:
	$(PYTHON) -I scripts/verify_inputs.py

test-python:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v

test-scripts:
	bash -n scripts/capture_admission.sh
	$(PYTHON) -I -c 'import ast,pathlib; [ast.parse(path.read_text()) for path in pathlib.Path("scripts").glob("*.py")]'

check: all verify-manifest test-reference test-reference-sanitize test-interlocks test-interlocks-sanitize verify-inputs test-self-check test-python test-scripts

offline-review:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I scripts/offline_review.py

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) tests/__pycache__ scripts/__pycache__
