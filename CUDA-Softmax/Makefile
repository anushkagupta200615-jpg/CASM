# ===========================================================================
#  Makefile -- builds the unified benchmark driver (all kernels linked in).
#
#  Pick your GPU arch:
#     make ARCH=sm_75   # Turing  (GTX 16xx, RTX 20xx, Colab/Kaggle T4)
#     make ARCH=sm_80   # Ampere datacenter (A100)
#     make ARCH=sm_86   # Ampere consumer (RTX 30-series)   <- default
#     make ARCH=sm_89   # Ada     (RTX 40-series)
#
#  Then:  ./softmax_bench [rows] [cols] [iters]
# ===========================================================================
ARCH    ?= sm_86
NVCC    ?= nvcc
NVFLAGS := -O3 -std=c++14 -arch=$(ARCH) -IHeader

TARGET  := softmax_bench
SRC     := utils/main.cu \
           utils/cpu_reference.cpp \
           Kernel/softmax_naive/softmax_naive.cu \
           Kernel/softmax_sharedmem/softmax_sharedmem.cu \
           Kernel/softmax_warpshuffle/softmax_warpshuffle.cu \
           Kernel/softmax_vectorized/softmax_vectorized.cu \
           Kernel/softmax_fused/softmax_fused.cu

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
