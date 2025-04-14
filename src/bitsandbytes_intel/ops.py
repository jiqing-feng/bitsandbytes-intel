from collections.abc import Sequence
import math

import torch

from .cpu_xpu_common import int8_linear_matmul_impl

print("Loading ops module")


def register_ops():
    print("Registering XPU implementations")

    # Check if the operator exists
    if not hasattr(torch.ops.bitsandbytes, "int8_linear_matmul"):
        raise RuntimeError("bitsandbytes::int8_linear_matmul not found! Make sure bitsandbytes is installed")

    @torch.library.impl("bitsandbytes::int8_linear_matmul", "XPU")
    def int8_linear_matmul_xpu(A: torch.Tensor, B: torch.Tensor):
        print("int8_linear_matmul_xpu called with tensors of shape:", A.shape, B.shape)
        return int8_linear_matmul_impl(A, B)

    @torch.library.impl("bitsandbytes::int8_linear_matmul.out", "XPU")
    def int8_linear_matmul_xpu_out(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor):
        print("int8_linear_matmul_xpu_out called with tensors of shape:", A.shape, B.shape)
        return int8_linear_matmul_impl(A, B, out)

    @torch.library.impl("bitsandbytes::dequantize_4bit.out", "XPU")
    def dequantize_4bit_xpu(
        A: torch.Tensor,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        shape: Sequence[int],
        dtype: torch.dtype,
        out: torch.Tensor,
    ) -> torch.Tensor:
        # TODO
        # if quant_type == "nf4" and getattr(quant_state, "ipex", False):
        #     output = torch.ops.torch_ipex.dequantize_4bit(A, "nf4", shape, absmax, None, blocksize).t()
        # else:
        #     output = dequantize_4bit_impl(A, quant_state, absmax, out, blocksize, quant_type)

        # return output
        raise NotImplementedError

    print("Successfully registered XPU implementation")

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


print("ops module loaded")
