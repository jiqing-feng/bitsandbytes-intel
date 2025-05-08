from collections.abc import Sequence
import math
from typing import Optional

import torch

from .xpu import (
    QuantState,
    _ipex_xpu_version_prereq,
    dequantize_4bit_impl,
    dequantize_blockwise_ipex_impl,
    dequantize_blockwise_torch_impl,
    gemv_4bit_impl,
    int8_linear_matmul_impl,
    int8_mm_dequant_impl,
    ipex_xpu,
    optimizer_update_8bit_blockwise,
    quantize_4bit_impl,
    quantize_blockwise_impl,
)

print("Loading ops module")


def register_xpu_ops():
    print("Registering XPU implementations")

    # Register the int8_linear_matmul implementation
    @torch.library.impl("bitsandbytes::int8_linear_matmul", "xpu")
    def int8_linear_matmul_xpu(A: torch.Tensor, B: torch.Tensor):
        return int8_linear_matmul_impl(A, B)

    @torch.library.impl("bitsandbytes::int8_linear_matmul.out", "xpu")
    def int8_linear_matmul_xpu_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
        return int8_linear_matmul_impl(A, B)

    # Register the int8_mm_dequant implementation
    @torch.library.impl("bitsandbytes::int8_mm_dequant", "xpu")
    def int8_mm_dequant_xpu(
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return int8_mm_dequant_impl(A, row_stats, col_stats, dtype, bias)

    # Register the quantize_4bit implementation
    @torch.library.impl("bitsandbytes::quantize_4bit", "xpu")
    def quantize_4bit_xpu(
        A: torch.Tensor,
        blocksize: int,
        quant_type: str,
        quant_storage: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_4bit_impl(
            A,
            blocksize,
            quant_type,
            quant_storage,
        )

    # Register the dequantize_4bit implementation
    @torch.library.impl("bitsandbytes::dequantize_4bit", "xpu")
    def dequantize_4bit_xpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        out = torch.empty(shape, dtype=dtype, device=A.device)
        return dequantize_4bit_impl(A, absmax, blocksize, quant_type, shape, dtype, out)

    # Register the quantize_blockwise implementation
    @torch.library.impl("bitsandbytes::quantize_blockwise", "xpu")
    def quantize_blockwise_xpu(
        A: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quantize_blockwise_impl(A, code, blocksize)

    # Register the dequantize_blockwise implementation
    dequantize_blockwise_impl = (
        dequantize_blockwise_ipex_impl if _ipex_xpu_version_prereq(2, 7) else dequantize_blockwise_torch_impl
    )

    @torch.library.impl("bitsandbytes::dequantize_blockwise", "xpu")
    def dequantize_blockwise_xpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        blocksize: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return dequantize_blockwise_impl(A, absmax, code, blocksize, dtype)

    # Register the gemv_4bit implementation
    @torch.library.impl("bitsandbytes::gemv_4bit", "xpu")
    def gemv_4bit_xpu(
        A: torch.Tensor,
        B: torch.Tensor,
        state: QuantState = None,
    ) -> torch.Tensor:
        return gemv_4bit_impl(A, B, state=state)

    # Register the optimizer_update_8bit_blockwise implementation
    @torch.library.impl("bitsandbytes::optimizer_update_8bit_blockwise", "xpu")
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
    # TODO: Need to check HPU
    elif hasattr(torch.backends, "hpu") and torch.backends.hpu.is_available():
        register_hpu_ops()


print("ops module loaded")
