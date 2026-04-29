import os
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _get_npu_aicore_num() -> int:
    device = torch.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    aicore_num = int(props.get("num_aicore", 0))
    if aicore_num <= 0:
        raise RuntimeError(f"Failed to query num_aicore from device properties: {props}")
    return aicore_num


_GENERIC_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 64}),
]

_FAST_CONFIGS = [
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 128}),
]

_B1_FAST_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
]


@triton.autotune(configs=_GENERIC_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_generic_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # One physical core handles multiple tiles in a cyclic manner.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = tl.cdiv(B, BLOCK_B)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_m_groups = tl.cdiv(M, GROUP_M)
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            if m_idx < M:
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)

                    a_ptrs = (
                        a_ptr
                        + offs_b[:, None] * stride_ab
                        + m_idx * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    w_ptrs = (
                        w_ptr
                        + m_idx * stride_wm
                        + offs_k[:, None] * stride_wk
                        + offs_n[None, :] * stride_wn
                    )

                    mask_a = (offs_b[:, None] < B) & (offs_k[None, :] < K)
                    mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                    w = tl.load(w_ptrs, mask=mask_w, other=0.0)

                    acc += tl.dot(a, w)

                c_ptrs = (
                    c_ptr
                    + offs_b[:, None] * stride_cb
                    + m_idx * stride_cm
                    + offs_n[None, :] * stride_cn
                )
                mask_c = (offs_b[:, None] < B) & (offs_n[None, :] < N)
                out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
                tl.store(c_ptrs, out, mask=mask_c)


@triton.autotune(configs=_FAST_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Fast kernel assumes no tail blocks in B/K/N and M divisible by GROUP_M.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = B // BLOCK_B
    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)

                a_ptrs = (
                    a_ptr
                    + offs_b[:, None] * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )

                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + offs_b[:, None] * stride_cb
                + m_idx * stride_cm
                + offs_n[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


@triton.autotune(configs=_B1_FAST_CONFIGS, key=["M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_b1_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Specialized no-mask kernel for B==1 to avoid masked lanes in tl.dot.
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_n_tiles

    offs_n = tl.arange(0, BLOCK_N)

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // num_n_tiles
        n_tile_idx = block_idx % num_n_tiles
        n_base = n_tile_idx * BLOCK_N
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a_ptrs = (
                    a_ptr
                    + 0 * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + (n_base + offs_n)[None, :] * stride_wn
                )
                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + 0 * stride_cb
                + m_idx * stride_cm
                + (n_base + offs_n)[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


def _to_nd_from_nz(tensor_b: torch.Tensor) -> torch.Tensor:
    # NZ input: [M, N/16, K, 16] -> ND [M, K, N]
    if tensor_b.ndim != 4:
        raise ValueError("NZ format expects tensor_b with 4 dimensions [M, N/16, K, 16].")
    if tensor_b.shape[-1] != 16:
        raise ValueError("NZ format expects tensor_b.shape[-1] == 16.")
    m, n16, k, inner = tensor_b.shape
    n = n16 * inner
    return tensor_b.permute(0, 2, 1, 3).contiguous().view(m, k, n)


def batch_matmul_transpose(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    format_mode: Optional[str] = None,
    quant_mode: Optional[str] = None,
) -> None:
    """
    Triton NPU implementation with the same public signature as:
      torch.ops.npu.batch_matmul_transpose(tensor_a, tensor_b, tensor_c, format_mode=None, quant_mode=None)

    Contract:
      tensor_a: [B, M, K]
      tensor_b (ND): [M, K, N]
      tensor_b (NZ): [M, N/16, K, 16]
      tensor_c: [B, M, N] (written in-place)
    """
    del quant_mode  # Kept for signature compatibility.

    mode = "ND" if format_mode is None else str(format_mode).upper()
    # Permanent fast path for decode hot loop:
    # - assume caller provides valid tensors
    # - avoid Python-side validation overhead
    if mode != "ND" and mode != "NZ":
        mode = "ND"

    if mode == "NZ":
        tensor_b_nd = _to_nd_from_nz(tensor_b)
    else:
        tensor_b_nd = tensor_b

    b, m, k = tensor_a.shape
    n = tensor_b_nd.shape[2]

    # Keep all pointers contiguous for predictable memory access on NPU.
    a = tensor_a.contiguous()
    w = tensor_b_nd.contiguous()

    # Ascend optimization for large logical grids.
    os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

    # Small-M strategy:
    # - M < 16: keep GROUP_M=1 to avoid reducing block count too aggressively.
    # - otherwise allow GROUP_M=2 when divisible to improve data reuse.
    group_m = 1 if m < 16 else (2 if (m % 2 == 0) else 1)

    # Fast path (mask-free): all fast configs currently assume BLOCK_B=8, BLOCK_N=128.
    # Keep divisibility guard strict to ensure no tail blocks.
    use_fast = (b % 8 == 0) and (n % 128 == 0) and (k % 128 == 0) and (m % group_m == 0)
    use_b1_fast = (b == 1) and (n % 128 == 0) and (k % 64 == 0) and (m % group_m == 0)

    grid = (_get_npu_aicore_num(),)
    if use_b1_fast:
        _batch_matmul_transpose_b1_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    elif use_fast:
        _batch_matmul_transpose_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    else:
        _batch_matmul_transpose_generic_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )


__all__ = ["batch_matmul_transpose"]
