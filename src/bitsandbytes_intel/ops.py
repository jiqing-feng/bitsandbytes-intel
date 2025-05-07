from collections.abc import Sequence
from typing import Optional
import math

import torch

from .cpu_xpu_common import (
    QuantState,
    int8_linear_matmul_impl,
    int8_double_quant_impl,
    int8_vectorwise_quant_impl,
    int8_mm_dequant_impl,
    quantize_4bit_impl,
    dequantize_4bit_impl,
    quantize_blockwise_impl,
    dequantize_blockwise_impl,
    gemm_4bit_impl,
    dequantize_blockwise_ipex_impl,
    optimizer_update_8bit_blockwise,
    ipex_xpu,
    ipex_cpu_only,
    _ipex_xpu_version_prereq,
)

print("Loading ops module")


def register_xpu_ops():
    print("Registering XPU implementations")

    # Register the int8_linear_matmul implementation
    @torch.library.impl("bitsandbytes::int8_linear_matmul", "XPU")
    def int8_linear_matmul_xpu(A: torch.Tensor, B: torch.Tensor):
       return int8_linear_matmul_impl(A, B)
    @torch.library.impl("bitsandbytes::int8_linear_matmul.out", "XPU")
    def int8_linear_matmul_xpu_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
        return int8_linear_matmul_impl(A, B, out)

    # Register the int8_double_quant implementation
    @torch.library.impl("bitsandbytes::int8_double_quant", "XPU")
    def int8_double_quant_xpu(
        A: torch.Tensor,
        threshold: float = 0.0,
        col_stats: torch.Tensor = None,
        row_stats: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_double_quant_impl(A, threshold, col_stats, row_stats)
    @torch.library.impl("bitsandbytes::int8_double_quant.out", "XPU")
    def int8_double_quant_xpu_out(
        A: torch.Tensor,
        threshold: float = 0.0,
        col_stats: torch.Tensor = None,
        row_stats: torch.Tensor = None,
        out_col: torch.Tensor = None,
        out_row: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_double_quant_impl(A, threshold, col_stats, row_stats, out_col, out_row)

    # Register the int8_vectorwise_quant implementation
    @torch.library.impl("bitsandbytes::int8_vectorwise_quant", "XPU")
    def int8_vectorwise_quant_xpu(
        A: torch.Tensor,
        threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_vectorwise_quant_impl(A, threshold)

    # Register the int8_mm_dequant implementation
    @torch.library.impl("bitsandbytes::int8_mm_dequant", "XPU")
    def int8_mm_dequant_xpu(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        bias: torch.Tensor = None,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ) -> torch.Tensor:
        return int8_mm_dequant_impl(A, row_stats, col_stats, bias, compute_dtype, output_dtype)
    @torch.library.impl("bitsandbytes::int8_mm_dequant.out", "XPU")
    def int8_mm_dequant_xpu_out(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        bias: torch.Tensor = None,
        compute_dtype = torch.float32,
        output_dtype = torch.float32,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return int8_mm_dequant_impl(A, row_stats, col_stats, bias, compute_dtype, output_dtype, out)

    # Register the quantize_4bit implementation
    @torch.library.impl("bitsandbytes::quantize_4bit", "XPU")
    def quantize_4bit_xpu(
        A: torch.Tensor,
        blocksize=64,
        quant_type="nf4",
        quant_storage=torch.uint8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_4bit_impl(
            A,
            blocksize,
            quant_type,
            quant_storage,
        )

    # Register the dequantize_4bit implementation
    @torch.library.impl("bitsandbytes::dequantize_4bit", "XPU")
    def dequantize_4bit_xpu(
        A: torch.Tensor,
        quant_state = None,
        absmax: torch.Tensor = None,
        blocksize: int = 64,
        quant_type = "nf4",
    ) -> torch.Tensor:
        return dequantize_4bit_impl(A, quant_state, absmax, blocksize, quant_type)
    @torch.library.impl("bitsandbytes::dequantize_4bit.out", "XPU")
    def dequantize_4bit_xpu_out(
        A: torch.Tensor,
        quant_state = None,
        absmax: torch.Tensor = None,
        blocksize: int = 64,
        quant_type = "nf4",
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return dequantize_4bit_impl(A, quant_state, absmax, blocksize, quant_type, out)

    # Register the quantize_blockwise implementation
    @torch.library.impl("bitsandbytes::quantize_blockwise", "XPU")
    def quantize_blockwise_xpu(
        A: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_blockwise_impl(A, code, blocksize)

    # Register the dequantize_blockwise implementation
    if _ipex_xpu_version_prereq(2, 7):
        dequantize_blockwise = dequantize_blockwise_ipex_impl
    else:
        dequantize_blockwise = dequantize_blockwise_impl

    @torch.library.impl("bitsandbytes::dequantize_blockwise", "XPU")
    def dequantize_blockwise_xpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return dequantize_blockwise(A, absmax, code, blocksize, dtype)
    @torch.library.impl("bitsandbytes::dequantize_blockwise.out", "XPU")
    def dequantize_blockwise_xpu_out(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return dequantize_blockwise(A, absmax, code, blocksize, dtype, out)

    # Register the gemv_4bit implementation
    @torch.library.impl("bitsandbytes::gemv_4bit", "XPU")
    def gemv_4bit_xpu(
        A: torch.Tensor,
        B: torch.Tensor,
        state: QuantState = None,
    ) -> torch.Tensor:
        return gemm_4bit_impl(A, B, state=state)
    @torch.library.impl("bitsandbytes::gemv_4bit.out", "XPU")
    def gemv_4bit_xpu_out(
        A: torch.Tensor,
        B: torch.Tensor,
        state: QuantState = None,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return gemm_4bit_impl(A, B, state=state, out=out)

    # Register the optimizer_update_8bit_blockwise implementation
    @torch.library.impl("bitsandbytes::optimizer_update_8bit_blockwise", "XPU")
    def optimizer_update_8bit_blockwise_xpu(
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
        state2: Optional[torch.Tensor],
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: torch.Tensor,
        qmap2: Optional[torch.Tensor],
        absmax1: torch.Tensor,
        absmax2: Optional[torch.Tensor],
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        skip_zeros=False,
    ) -> None:
        optimizer_update_8bit_blockwise(
            optimizer_name,
            g,
            p,
            state1,
            state2,
            beta1,
            beta2,
            beta3,
            alpha,
            eps,
            step,
            lr,
            qmap1,
            qmap2,
            absmax1,
            absmax2,
            weight_decay,
            gnorm_scale,
            skip_zeros,
        )

    print("Successfully registered XPU implementation")


def register_cpu_ops():
    print("Registering CPU implementations")

    # Register the int8_linear_matmul implementation
    @torch.library.impl("bitsandbytes::int8_linear_matmul", "CPU")
    def int8_linear_matmul_cpu(A: torch.Tensor, B: torch.Tensor):
        return int8_linear_matmul_impl(A, B)
    @torch.library.impl("bitsandbytes::int8_linear_matmul.out", "CPU")
    def int8_linear_matmul_cpu_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
        return int8_linear_matmul_impl(A, B, out)

    # Register the int8_double_quant implementation
    @torch.library.impl("bitsandbytes::int8_double_quant", "CPU")
    def int8_double_quant_cpu(
        A: torch.Tensor,
        threshold: float = 0.0,
        col_stats: torch.Tensor = None,
        row_stats: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_double_quant_impl(A, threshold, col_stats, row_stats)
    @torch.library.impl("bitsandbytes::int8_double_quant.out", "CPU")
    def int8_double_quant_cpu_out(
        A: torch.Tensor,
        threshold: float = 0.0,
        col_stats: torch.Tensor = None,
        row_stats: torch.Tensor = None,
        out_col: torch.Tensor = None,
        out_row: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_double_quant_impl(A, threshold, col_stats, row_stats, out_col, out_row)

    # Register the int8_vectorwise_quant implementation
    @torch.library.impl("bitsandbytes::int8_vectorwise_quant", "CPU")
    def int8_vectorwise_quant_cpu(
        A: torch.Tensor,
        threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return int8_vectorwise_quant_impl(A, threshold)

    # Register the int8_mm_dequant implementation
    @torch.library.impl("bitsandbytes::int8_mm_dequant", "CPU")
    def int8_mm_dequant_cpu(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        bias: torch.Tensor = None,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    ) -> torch.Tensor:
        return int8_mm_dequant_impl(A, row_stats, col_stats, bias, compute_dtype, output_dtype)
    @torch.library.impl("bitsandbytes::int8_mm_dequant.out", "CPU")
    def int8_mm_dequant_cpu_out(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        bias: torch.Tensor = None,
        compute_dtype = torch.float32,
        output_dtype = torch.float32,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return int8_mm_dequant_impl(A, row_stats, col_stats, bias, compute_dtype, output_dtype, out)

    # Register the quantize_4bit implementation
    @torch.library.impl("bitsandbytes::quantize_4bit", "CPU")
    def quantize_4bit_cpu(
        A: torch.Tensor,
        blocksize=64,
        quant_type="nf4",
        quant_storage=torch.uint8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_4bit_impl(
            A,
            blocksize,
            quant_type,
            quant_storage,
        )

    # Register the dequantize_4bit implementation
    @torch.library.impl("bitsandbytes::dequantize_4bit", "CPU")
    def dequantize_4bit_cpu(
        A: torch.Tensor,
        quant_state = None,
        absmax: torch.Tensor = None,
        blocksize: int = 64,
        quant_type = "nf4",
    ) -> torch.Tensor:
        return dequantize_4bit_impl(A, quant_state, absmax, blocksize, quant_type)
    @torch.library.impl("bitsandbytes::dequantize_4bit.out", "CPU")
    def dequantize_4bit_cpu_out(
        A: torch.Tensor,
        quant_state = None,
        absmax: torch.Tensor = None,
        blocksize: int = 64,
        quant_type = "nf4",
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return dequantize_4bit_impl(A, quant_state, absmax, blocksize, quant_type, out)

    # Register the quantize_blockwise implementation
    @torch.library.impl("bitsandbytes::quantize_blockwise", "CPU")
    def quantize_blockwise_cpu(
        A: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_blockwise_impl(A, code, blocksize)

    # Register the dequantize_blockwise implementation
    @torch.library.impl("bitsandbytes::dequantize_blockwise", "CPU")
    def dequantize_blockwise_cpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return dequantize_blockwise_impl(A, absmax, code, blocksize, dtype)
    @torch.library.impl("bitsandbytes::dequantize_blockwise.out", "CPU")
    def dequantize_blockwise_cpu_out(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return dequantize_blockwise_impl(A, absmax, code, blocksize, dtype, out)

    # Register the gemv_4bit implementation
    @torch.library.impl("bitsandbytes::gemv_4bit", "CPU")
    def gemv_4bit_cpu(
        A: torch.Tensor,
        B: torch.Tensor,
        state: QuantState = None,
    ) -> torch.Tensor:
        return gemm_4bit_impl(A, B, state=state)
    @torch.library.impl("bitsandbytes::gemv_4bit.out", "CPU")
    def gemv_4bit_cpu_out(
        A: torch.Tensor,
        B: torch.Tensor,
        state: QuantState = None,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        return gemm_4bit_impl(A, B, state=state, out=out)

    print("Successfully registered CPU implementation")


def register_hpu_ops():
    print("Registering HPU implementations")

    @torch.library.impl("bitsandbytes::dequantize_4bit", "HPU")
    def dequantize_4bit_hpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        out_shape = (math.prod(shape),)
        out_dq = torch.ops.hpu.dequantize_nf4(
            input,
            absmax,
            blocksize,
            out_shape=out_shape,
            out_dtype=dtype,
        )
        output = out_dq.reshape(shape).T
        return output

    @torch.library.impl("bitsandbytes::quantize_4bit", "HPU")
    def quantize_4bit_hpu(
        A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    print("Successfully registered HPU implementations")


def register_ops():
    # Check if the operator exists
    if not hasattr(torch.ops.bitsandbytes, "int8_linear_matmul"):
        raise RuntimeError("bitsandbytes::int8_linear_matmul not found! Make sure bitsandbytes is installed")

    if ipex_xpu:
        register_xpu_ops()
    elif ipex_cpu_only:
        register_cpu_ops()
    # TODO: Need to check HPU
    else:
        register_hpu_ops()


print("ops module loaded")
