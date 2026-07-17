# ===========================================================================
#  Makefile
#     make            -> builds both benchmarks (forward + backward)
#     make forward    -> ./softmax_bench           (softmax forward pass)
#     make backward   -> ./softmax_backward_bench  (softmax gradient / training)
#     make run        -> build + run the forward benchmark
#     make run-bwd    -> build + run the backward benchmark
#     make clean
#
#  Pick your GPU arch:
#     make ARCH=sm_75   # Turing  (GTX 16xx, RTX 20xx, Colab/Kaggle T4)
#     make ARCH=sm_80   # Ampere datacenter (A100)
#     make ARCH=sm_86   # Ampere consumer (RTX 30-series)   <- default
#     make ARCH=sm_89   # Ada     (RTX 40-series)
# ===========================================================================
ARCH    ?= sm_86
NVCC    ?= nvcc
NVFLAGS := -O3 -std=c++14 -arch=$(ARCH) -IHeader

FWD_BIN := softmax_bench
BWD_BIN := softmax_backward_bench

COMMON  := utils/cpu_reference.cpp \
           Kernel/softmax_naive/softmax_naive.cu \
           Kernel/softmax_sharedmem/softmax_sharedmem.cu \
           Kernel/softmax_warpshuffle/softmax_warpshuffle.cu \
           Kernel/softmax_vectorized/softmax_vectorized.cu \
           Kernel/softmax_fused/softmax_fused.cu \
           Kernel/softmax_backward/softmax_backward.cu

FWD_SRC := utils/main.cu          $(COMMON)
BWD_SRC := utils/backward_main.cu $(COMMON)

.PHONY: all forward backward run run-bwd clean
all: forward backward

forward: $(FWD_BIN)
backward: $(BWD_BIN)

$(FWD_BIN): $(FWD_SRC)
	$(NVCC) $(NVFLAGS) $(FWD_SRC) -o $(FWD_BIN)

$(BWD_BIN): $(BWD_SRC)
	$(NVCC) $(NVFLAGS) $(BWD_SRC) -o $(BWD_BIN)

run: $(FWD_BIN)
	./$(FWD_BIN)

run-bwd: $(BWD_BIN)
	./$(BWD_BIN)

clean:
	rm -f $(FWD_BIN) $(BWD_BIN)
