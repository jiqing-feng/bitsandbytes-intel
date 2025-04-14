# `bitsandbytes` Intel Backend

Registration for Intel optimized bitsandbytes operators.

## Quick Start

```
# Build and enter container
docker compose run --build --rm bnb-intel-dev /bin/bash

# Run validation (inside container):
python -m bitsandbytes_intel
```

## Testing

Expected successful output:
```
root@pvc-hf-1100-00:/workspace# python -m bnb_intel
Initializing bnb_intel module
[W414 18:23:28.291667720 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
2025-04-14 18:23:29,577 - bitsandbytes.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
Loading ops module
ops module loaded
Registering XPU implementations
Successfully registered XPU implementation
Registering HPU implementations
Successfully registered HPU implementations
🧪 Running minimal XPU backend test...
int8_linear_matmul_xpu called with tensors of shape: torch.Size([32, 64]) torch.Size([128, 64])

✅ Operator executed successfully!
   Input shapes: torch.Size([32, 64]) x torch.Size([128, 64])
   Output shape: torch.Size([32, 128])
   Output device: xpu:0
[W414 18:23:30.825181068 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
```

## Technical Implementation

Key files:
- `src/bitsandbytes_intel/ops.py` - Intel kernel registration
- `src/bitsandbytes_intel/__init__.py` - Autoload setup
- `docker-compose.yml` - Build environment
- `setup.py` - Package configuration

Uses PyTorch's autoload mechanism to register:
```
@torch.library.impl("bitsandbytes::int8_linear_matmul", "XPU")
```
