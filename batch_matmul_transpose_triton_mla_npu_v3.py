from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - fallback is handled at runtime.
    triton = None
    tl = None


_GENERIC_MAIN_BLOCK_B = 8
_GENERIC_MAIN_BLOCK_N = 128
_GENERIC_MAIN_BLOCK_K = 64


@lru_cache(maxsize=1)
def _get_num_aicore() -> int:
    if triton is None:
        raise RuntimeError("Triton is not available.")
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu is required to query Ascend device properties.")

    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(device)
    )
    num_aicore = int(device_properties.get("num_aicore", 0))
    if num_aicore <= 0:
        raise RuntimeError("Failed to detect num_aicore from the active Ascend device.")
    return num_aicore


def _is_fast_path_candidate(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
) -> bool:
    return (
        tensor_a.is_contiguous()
        and tensor_b.is_contiguous()
        and tensor_c.is_contiguous()
        and tensor_a.shape[0] % 16 == 0
        and tensor_a.shape[-1] % 128 == 0
        and tensor_b.shape[-1] % 128 == 0
    )


def _can_use_split_generic_main(batch_size: int, k: int, n: int) -> bool:
    return (
        batch_size >= _GENERIC_MAIN_BLOCK_B
        and n >= _GENERIC_MAIN_BLOCK_N
        and k % _GENERIC_MAIN_BLOCK_K == 0
    )


if triton is not None:

    _STRICT32_SMALL_HEAD_CONFIGS = [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
    ]
    _STRICT32_CONFIGS = [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}),
    ]
    _STRICT16_CONFIGS = [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}),
    ]
    _GENERIC_TAIL_CONFIGS = [
        triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 8, "BLOCK_N": 256, "BLOCK_K": 64}),
        triton.Config({"BLOCK_B": 4, "BLOCK_N": 128, "BLOCK_K": 64}),
    ]

    @triton.autotune(
        configs=_STRICT32_SMALL_HEAD_CONFIGS, key=["batch_size", "num_heads", "k", "n"]
    )
    @triton.jit
    def _batch_matmul_transpose_strict32_small_head_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        batch_size,
        num_heads,
        k,
        n,
        stride_ab,
        stride_ah,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        BLOCK_B: tl.constexpr = 32

        num_n_blocks = tl.cdiv(n, BLOCK_N)
        total_tasks = num_heads * num_n_blocks

        offs_b = tl.arange(0, BLOCK_B)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        for task_id in range(pid, total_tasks, num_core):
            head_id = task_id // num_n_blocks
            n_block_id = task_id % num_n_blocks
            n_start = n_block_id * BLOCK_N

            head_offset = head_id.to(tl.int64)
            n_offsets = (n_start + offs_n).to(tl.int64)

            for b_start in range(0, batch_size, BLOCK_B):
                b_offsets = (b_start + offs_b).to(tl.int64)
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, k, BLOCK_K):
                    k_offsets = (k_start + offs_k).to(tl.int64)
                    a_ptrs = (
                        a_ptr
                        + b_offsets[:, None] * stride_ab
                        + head_offset * stride_ah
                        + k_offsets[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + head_offset * stride_bh
                        + k_offsets[:, None] * stride_bk
                        + n_offsets[None, :] * stride_bn
                    )

                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc += tl.dot(a, b)

                c_ptrs = (
                    c_ptr
                    + b_offsets[:, None] * stride_cb
                    + head_offset * stride_ch
                    + n_offsets[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs,
                    acc.to(c_ptr.dtype.element_ty),
                )

    @triton.autotune(configs=_STRICT32_CONFIGS, key=["batch_size", "num_heads", "k", "n"])
    @triton.jit
    def _batch_matmul_transpose_strict32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        batch_size,
        num_heads,
        k,
        n,
        stride_ab,
        stride_ah,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        BLOCK_B: tl.constexpr = 32

        num_n_blocks = tl.cdiv(n, BLOCK_N)
        total_tasks = num_heads * num_n_blocks

        offs_b = tl.arange(0, BLOCK_B)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        for task_id in range(pid, total_tasks, num_core):
            head_id = task_id // num_n_blocks
            n_block_id = task_id % num_n_blocks
            n_start = n_block_id * BLOCK_N

            head_offset = head_id.to(tl.int64)
            n_offsets = (n_start + offs_n).to(tl.int64)

            for b_start in range(0, batch_size, BLOCK_B):
                b_offsets = (b_start + offs_b).to(tl.int64)
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, k, BLOCK_K):
                    k_offsets = (k_start + offs_k).to(tl.int64)
                    a_ptrs = (
                        a_ptr
                        + b_offsets[:, None] * stride_ab
                        + head_offset * stride_ah
                        + k_offsets[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + head_offset * stride_bh
                        + k_offsets[:, None] * stride_bk
                        + n_offsets[None, :] * stride_bn
                    )

                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc += tl.dot(a, b)

                c_ptrs = (
                    c_ptr
                    + b_offsets[:, None] * stride_cb
                    + head_offset * stride_ch
                    + n_offsets[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs,
                    acc.to(c_ptr.dtype.element_ty),
                )

    @triton.autotune(configs=_STRICT16_CONFIGS, key=["batch_size", "num_heads", "k", "n"])
    @triton.jit
    def _batch_matmul_transpose_strict16_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        batch_size,
        num_heads,
        k,
        n,
        stride_ab,
        stride_ah,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cn,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)
        BLOCK_B: tl.constexpr = 16

        num_n_blocks = tl.cdiv(n, BLOCK_N)
        total_tasks = num_heads * num_n_blocks

        offs_b = tl.arange(0, BLOCK_B)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        for task_id in range(pid, total_tasks, num_core):
            head_id = task_id // num_n_blocks
            n_block_id = task_id % num_n_blocks
            n_start = n_block_id * BLOCK_N

            head_offset = head_id.to(tl.int64)
            n_offsets = (n_start + offs_n).to(tl.int64)

            for b_start in range(0, batch_size, BLOCK_B):
                b_offsets = (b_start + offs_b).to(tl.int64)
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, k, BLOCK_K):
                    k_offsets = (k_start + offs_k).to(tl.int64)
                    a_ptrs = (
                        a_ptr
                        + b_offsets[:, None] * stride_ab
                        + head_offset * stride_ah
                        + k_offsets[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + head_offset * stride_bh
                        + k_offsets[:, None] * stride_bk
                        + n_offsets[None, :] * stride_bn
                    )

                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc += tl.dot(a, b)

                c_ptrs = (
                    c_ptr
                    + b_offsets[:, None] * stride_cb
                    + head_offset * stride_ch
                    + n_offsets[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs,
                    acc.to(c_ptr.dtype.element_ty),
                )

    @triton.jit
    def _batch_matmul_transpose_generic_main_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        batch_size,
        num_heads,
        k,
        n,
        stride_ab,
        stride_ah,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cn,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)

        batch_main = (batch_size // BLOCK_B) * BLOCK_B
        num_n_blocks = n // BLOCK_N
        total_tasks = num_heads * num_n_blocks

        offs_b = tl.arange(0, BLOCK_B)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        for task_id in range(pid, total_tasks, num_core):
            head_id = task_id // num_n_blocks
            n_block_id = task_id % num_n_blocks
            n_start = n_block_id * BLOCK_N

            head_offset = head_id.to(tl.int64)
            n_offsets = (n_start + offs_n).to(tl.int64)

            for b_start in range(0, batch_main, BLOCK_B):
                b_offsets = (b_start + offs_b).to(tl.int64)
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, k, BLOCK_K):
                    k_offsets = (k_start + offs_k).to(tl.int64)
                    a_ptrs = (
                        a_ptr
                        + b_offsets[:, None] * stride_ab
                        + head_offset * stride_ah
                        + k_offsets[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + head_offset * stride_bh
                        + k_offsets[:, None] * stride_bk
                        + n_offsets[None, :] * stride_bn
                    )

                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc += tl.dot(a, b)

                c_ptrs = (
                    c_ptr
                    + b_offsets[:, None] * stride_cb
                    + head_offset * stride_ch
                    + n_offsets[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs,
                    acc.to(c_ptr.dtype.element_ty),
                )

    @triton.autotune(
        configs=_GENERIC_TAIL_CONFIGS, key=["batch_size", "num_heads", "k", "n"]
    )
    @triton.jit
    def _batch_matmul_transpose_generic_tail_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        batch_size,
        num_heads,
        k,
        n,
        stride_ab,
        stride_ah,
        stride_ak,
        stride_bh,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_ch,
        stride_cn,
        main_batch_limit,
        main_n_limit,
        SKIP_MAIN_TILES: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_core = tl.num_programs(0)

        num_n_blocks = tl.cdiv(n, BLOCK_N)
        total_tasks = num_heads * num_n_blocks

        offs_b = tl.arange(0, BLOCK_B)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        for task_id in range(pid, total_tasks, num_core):
            head_id = task_id // num_n_blocks
            n_block_id = task_id % num_n_blocks
            n_start = n_block_id * BLOCK_N

            head_offset = head_id.to(tl.int64)
            n_offsets = (n_start + offs_n).to(tl.int64)
            n_mask = n_offsets < n

            for b_start in range(0, batch_size, BLOCK_B):
                if (
                    SKIP_MAIN_TILES
                    and (b_start + BLOCK_B) <= main_batch_limit
                    and (n_start + BLOCK_N) <= main_n_limit
                ):
                    continue

                b_offsets = (b_start + offs_b).to(tl.int64)
                b_mask = b_offsets < batch_size
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k_start in range(0, k, BLOCK_K):
                    k_offsets = (k_start + offs_k).to(tl.int64)
                    k_mask = k_offsets < k

                    a_ptrs = (
                        a_ptr
                        + b_offsets[:, None] * stride_ab
                        + head_offset * stride_ah
                        + k_offsets[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + head_offset * stride_bh
                        + k_offsets[:, None] * stride_bk
                        + n_offsets[None, :] * stride_bn
                    )

                    a = tl.load(a_ptrs, mask=b_mask[:, None] & k_mask[None, :], other=0.0)
                    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                    acc += tl.dot(a, b)

                c_ptrs = (
                    c_ptr
                    + b_offsets[:, None] * stride_cb
                    + head_offset * stride_ch
                    + n_offsets[None, :] * stride_cn
                )
                tl.store(
                    c_ptrs,
                    acc.to(c_ptr.dtype.element_ty),
                    mask=b_mask[:, None] & n_mask[None, :],
                )


def batch_matmul_transpose(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, k = tensor_a.shape
    n = tensor_b.shape[2]

    num_aicore = _get_num_aicore()
    grid = (num_aicore,)

    if _is_fast_path_candidate(tensor_a, tensor_b, tensor_c):
        if batch_size % 32 == 0:
            kernel = (
                _batch_matmul_transpose_strict32_small_head_kernel
                if num_heads <= 8
                else _batch_matmul_transpose_strict32_kernel
            )
        else:
            kernel = _batch_matmul_transpose_strict16_kernel

        kernel[grid](
            tensor_a,
            tensor_b,
            tensor_c,
            batch_size,
            num_heads,
            k,
            n,
            tensor_a.stride(0),
            tensor_a.stride(1),
            tensor_a.stride(2),
            tensor_b.stride(0),
            tensor_b.stride(1),
            tensor_b.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
        )
        return tensor_c

    use_split_generic_main = _can_use_split_generic_main(batch_size, k, n)
    main_batch_limit = (
        (batch_size // _GENERIC_MAIN_BLOCK_B) * _GENERIC_MAIN_BLOCK_B
        if use_split_generic_main
        else 0
    )
    main_n_limit = (
        (n // _GENERIC_MAIN_BLOCK_N) * _GENERIC_MAIN_BLOCK_N if use_split_generic_main else 0
    )

    if use_split_generic_main:
        _batch_matmul_transpose_generic_main_kernel[grid](
            tensor_a,
            tensor_b,
            tensor_c,
            batch_size,
            num_heads,
            k,
            n,
            tensor_a.stride(0),
            tensor_a.stride(1),
            tensor_a.stride(2),
            tensor_b.stride(0),
            tensor_b.stride(1),
            tensor_b.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            BLOCK_B=_GENERIC_MAIN_BLOCK_B,
            BLOCK_N=_GENERIC_MAIN_BLOCK_N,
            BLOCK_K=_GENERIC_MAIN_BLOCK_K,
        )

    _batch_matmul_transpose_generic_tail_kernel[grid](
        tensor_a,
        tensor_b,
        tensor_c,
        batch_size,
        num_heads,
        k,
        n,
        tensor_a.stride(0),
        tensor_a.stride(1),
        tensor_a.stride(2),
        tensor_b.stride(0),
        tensor_b.stride(1),
        tensor_b.stride(2),
        tensor_c.stride(0),
        tensor_c.stride(1),
        tensor_c.stride(2),
        main_batch_limit,
        main_n_limit,
        SKIP_MAIN_TILES=use_split_generic_main,
    )
    return tensor_c
