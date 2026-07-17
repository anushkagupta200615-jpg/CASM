#!/usr/bin/env python3
"""
Production baseline: time PyTorch's native softmax (cuDNN under the hood) on the
exact same shape you benchmark the CUDA kernels with, so the comparison is fair.

    python bench/benchmark_pytorch.py --rows 4096 --cols 4096 --iters 100

Copy the printed "ms/call" into the README table's "PyTorch (cuDNN)" row.
"""
import argparse
import time

import torch


def bench(rows: int, cols: int, iters: int) -> float:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible to PyTorch. Run this on a GPU box / Colab.")

    dev = torch.device("cuda")
    x = torch.randn(rows, cols, device=dev)

    # warmup
    for _ in range(10):
        torch.softmax(x, dim=1)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        torch.softmax(x, dim=1)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms per call


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shape : {args.rows} x {args.cols}   iters={args.iters}")
    ms = bench(args.rows, args.cols, args.iters)
    print(f"PyTorch softmax (cuDNN): {ms:.4f} ms/call")


if __name__ == "__main__":
    main()
