from collections.abc import Sequence
import subprocess
from typing import Optional
import warnings

import torch
import torch.nn.functional as F

try:
    # to support Intel CPU/GPU (XPU) backend
    import intel_extension_for_pytorch as ipex

    ipex_cpu = ipex if ipex._C._has_cpu() else None
    ipex_xpu = ipex if ipex._C._has_xpu() else None
    ipex_cpu_only = ipex._C._has_cpu() and (not ipex._C._has_xpu())
except BaseException:
    ipex_cpu = None
    ipex_xpu = None
    ipex_cpu_only = None


gxx_available = False
try:
    subprocess.run(["g++", "--version"], capture_output=True)  # hide terminal output
    gxx_available = True
except BaseException:
    warnings.warn("g++ not found, torch.compile disabled for CPU/XPU.")


Tensor = torch.Tensor


def _torch_version_prereq(major, minor):
    ver_major = int(torch.__version__.split(".")[0])
    ver_minor = int(torch.__version__.split(".")[1])
    return ver_major * 32 + ver_minor >= major * 32 + minor


def _ipex_xpu_version_prereq(major, minor):
    if ipex_xpu is not None:
        ver_major = ipex_xpu.__version__.split(".")[0]
        ver_minor = ipex_xpu.__version__.split(".")[1]
        return int(ver_major) * 32 + int(ver_minor) >= major * 32 + minor
    return False


str2optimizer8bit_blockwise = {}
if ipex_xpu is not None and _ipex_xpu_version_prereq(2, 7):
    str2optimizer8bit_blockwise = {
        "adam": (
            ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_fp32,
            ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_fp16,
            ipex.xpu.bitsandbytes.cadam_8bit_blockwise_grad_bf16,
        ),
    }


def _maybe_torch_compile(func):
    # torch.compile requires g++ and pytorch >= 2.0
    if gxx_available and _torch_version_prereq(2, 0) and not ipex_xpu:
        options = {}
        # fx_graph_cache requires pytorch >= 2.2
        if _torch_version_prereq(2, 2):
            options.update({"fx_graph_cache": True})
        return torch.compile(func, dynamic=True, options=options)
    return func


def transform(
    A: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    transpose=False,
    state: Optional[tuple[torch.Size, str]] = None,
):
    """
    Transform tensor A to to_order. It is originally designed for CUDA.
    For CPU/XPU, it returns the original tensor if transpose=False.
    Otherwise, it returns the transpose of A
    """
    if transpose:
        if out is not None:
            out.copy_(A.T)
        else:
            out = A.T
    else:
        if out is not None:
            out.copy_(A)
        else:
            out = A
    return out, state


# Applied from cpu int8_linear_matmul op
def int8_linear_matmul_impl(A: torch.Tensor, B: torch.Tensor):
    return torch._int_mm(
        A.reshape(-1, A.shape[-1]),
        B.t(),
    ).reshape(*A.shape[:-1], B.shape[0])


@_maybe_torch_compile
def int8_mm_dequant_impl(
    A: torch.Tensor,
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    torch._check(A.dtype == torch.int32, lambda: f"A must be int32, got {A.dtype}")
    torch._check(row_stats.dtype == torch.float32, lambda: f"row_stats must be float32, got {row_stats.dtype}")
    torch._check(col_stats.dtype == torch.float32, lambda: f"col_stats must be float32, got {col_stats.dtype}")

    A_calc = A.view(-1, A.shape[-1])
    row_stats = row_stats.reshape(-1).unsqueeze(-1)
    col_stats = col_stats.reshape(-1).unsqueeze(0)

    out = A_calc * (row_stats * col_stats) * 6.200124e-05
    if bias is not None:
        out += bias

    return out.to(dtype or torch.float16)


_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
    device="xpu" if torch.xpu.is_available() else "cpu",
)
_FP4_QUANT_TABLE = torch.tensor(
    [
        0.0000,
        0.0052,
        0.6667,
        1.0000,
        0.3333,
        0.5000,
        0.1667,
        0.2500,
        0.0000,
        -0.0052,
        -0.6667,
        -1.0000,
        -0.3333,
        -0.5000,
        -0.1667,
        -0.2500,
    ],
    dtype=torch.float32,
    device="xpu" if torch.xpu.is_available() else "cpu",
)
CODE = {"nf4": _NF4_QUANT_TABLE, "fp4": _FP4_QUANT_TABLE}


def quantize_blockwise_impl(
    A: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor A in blocks of 8-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized to int8.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization code.
    blocksize : int
        The blocksize used in quantization.

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor with packed 4-bit values.
    torch.Tensor:
        The absmax.
    """
    n = A.numel()
    rem = n % blocksize
    has_rem = rem > 0
    blocks = n // blocksize + has_rem
    absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

    diff = torch.abs(scaled_A.unsqueeze(-1) - code.to(scaled_A.device))
    out_uint8 = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device).reshape(A.shape)

    return out_uint8, absmax


def dequantize_blockwise_torch_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert A.dtype == torch.uint8
    out = code[A.reshape(-1).int()]
    blocks = out.shape[-1] // blocksize
    res = out.shape[-1] % blocksize
    if res != 0:
        out = F.pad(out, (0, blocksize - res), mode="constant", value=0)
    out = (out.view(-1, blocksize) * absmax.view(-1, 1)).to(dtype).reshape(-1)
    out = out[: blocks * blocksize + res]
    out = out.reshape(A.shape)

    return out


# Currently only works for XPU
def dequantize_blockwise_ipex_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if ipex_xpu is None or not _ipex_xpu_version_prereq(2, 7):
        raise RuntimeError("Please install intel_extension_for_ipex >= 2.7 for 8bit optimizer backend on XPU device.")

    out = torch.empty(A.reshape(-1).shape, dtype=dtype, device=A.device)
    # void cdequantize_blockwise_fp32(
    # float *code, unsigned char *A, float *absmax, float *out, int blocksize, const int n, cudaStream_t stream)
    if dtype == torch.float16:
        ipex.xpu.bitsandbytes.cdequantize_blockwise_fp16(code, A, absmax, out, blocksize, A.numel())
    elif dtype == torch.bfloat16:
        ipex.xpu.bitsandbytes.cdequantize_blockwise_bf16(code, A, absmax, out, blocksize, A.numel())
    elif dtype == torch.float32:
        ipex.xpu.bitsandbytes.cdequantize_blockwise_fp32(code, A, absmax, out, blocksize, A.numel())
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")

    return out


# Copied from cpu quantize_4bit op
def quantize_4bit_impl(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()
    rem = n % blocksize
    has_rem = rem > 0
    blocks = n // blocksize + has_rem

    # Scale tensor to [-1, 1]
    absmax = torch.zeros((blocks,), device=A.device, dtype=A.dtype)
    A_reshaped = A.reshape(n)
    A_com_reshaped = A_reshaped[: n - rem].reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled = torch.clamp(A_com_reshaped * (1 / absmax[: blocks - has_rem].view(-1, 1)), -1, 1).reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_rem = torch.clamp(A_reshaped[n - rem :] * (1 / absmax[-1]), -1, 1)
        scaled = torch.cat([scaled, scaled_rem], dim=0)
    # Quantize with the lookup table
    quant_table = CODE[quant_type].to(scaled.device)
    quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - quant_table), dim=-1, keepdim=True).to(torch.uint8)

    # Pack two quantized values per byte
    packed = quantized[::2] << 4 | quantized[1::2]

    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


# Copied from cpu dequantize_4bit op
# Compile will fail in torch.frombuffer
# @_maybe_torch_compile
def dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> Tensor:
    torch._check_is_size(blocksize)
    torch._check(quant_type in ("nf4", "fp4"), lambda: f"quant_type must be nf4 or fp4 on CPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    # Enable non uint8 dtype
    device = A.device
    if A.dtype != torch.uint8:
        if A.dtype == torch.bfloat16:
            # Numpy does not support bfloat16
            A = A.view(torch.float16)
        bytes_value = A.cpu().numpy().tobytes()
        A = torch.frombuffer(bytes_value, dtype=torch.uint8).to(device)

    A = A.reshape(-1)
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    out_dq[1::2] = A & 0xF
    out_dq[::2] = A >> 4
    # code is fp32, cast to dtype to avoid the mismatch issue
    code = CODE[quant_type].to(out_dq.device).to(dtype)
    out_dq = code[out_dq]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)

    rem = n % blocksize
    has_rem = rem > 0
    blocks = n // blocksize + has_rem

    out = torch.empty(shape, dtype=dtype, device=A.device).reshape(-1)
    if has_rem:
        out[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
        out[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

    out = out.reshape(-1, *shape[1:]).to(dtype)

    return out


# Copied from cpu gemv_4bit op
def gemv_4bit_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    # Applied from dequantize_4bit
    quant_type = "nf4" if code[1] > 0 else "fp4"
    B_dq = dequantize_4bit_impl(B, absmax, blocksize, quant_type, shapeB, A.dtype)

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()

    return torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )


def optimizer_update_8bit_blockwise(
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
    optim_func = None
    if ipex_xpu is None or not _ipex_xpu_version_prereq(2, 7):
        raise RuntimeError("Please install intel_extension_for_ipex >= 2.7 for 8bit optimizer backend on XPU device.")

    if g.dtype == torch.float32 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][0]
    elif g.dtype == torch.float16 and state1.dtype == torch.uint8:
        optim_func = str2optimizer8bit_blockwise[optimizer_name][1]
    elif (
        g.dtype == torch.bfloat16
        and state1.dtype == torch.uint8
        and len(str2optimizer8bit_blockwise[optimizer_name]) == 3
    ):
        optim_func = str2optimizer8bit_blockwise[optimizer_name][2]
    else:
        raise ValueError(
            f"Gradient+optimizer bit data type combination not supported: grad {g.dtype}, optimizer {state1.dtype}",
        )
    optim_func(
        p,
        g,
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
        g.numel(),
    )


def optimizer_update_32bit(
    optimizer_name: str,
    g: torch.Tensor,
    p: torch.Tensor,
    state1: torch.Tensor,
    beta1: float,
    eps: float,
    step: int,
    lr: float,
    state2: Optional[torch.Tensor] = None,
    beta2: float = 0.0,
    beta3: float = 0.0,
    alpha: float = 0.0,
    weight_decay: float = 0.0,
    gnorm_scale: float = 1.0,
    unorm_vec: Optional[torch.Tensor] = None,
    max_unorm: float = 0.0,
    skip_zeros=False,
) -> None:
    raise NotImplementedError
