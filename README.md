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
bnb_intel module initialization complete, _autoload = <function _autoload at 0x7f552f6cb1c0>
[W318 10:30:23.240989074 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
2025-03-18 10:30:24,784 - bitsandbytes.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
Loading ops module
ops module loaded
Registering XPU implementations
Successfully registered XPU implementation
🧪 Running minimal XPU backend test...
int8_linear_matmul_xpu called with tensors of shape: torch.Size([32, 64]) torch.Size([64, 128])

✅ Operator executed successfully!
   Input shapes: torch.Size([32, 64]) x torch.Size([64, 128])
   Output shape: torch.Size([32, 128])
   Output device: xpu:0
[W318 10:30:25.828258019 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
root@pvc-hf-1100-00:/workspace# Initializing bnb_intel module
bnb_intel module initialization complete, _autoload = <function _autoload at 0x7fb7851f6a70>
2025-03-18 10:30:27,965 - bitsandbytes.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
Loading ops module
ops module loaded
Registering XPU implementations
Successfully registered XPU implementation

root@pvc-hf-1100-00:/workspace# python -m bnb_intel
Initializing bnb_intel module
bnb_intel module initialization complete, _autoload = <function _autoload at 0x7f874aaf31c0>
[W318 10:30:59.381272647 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
2025-03-18 10:31:00,902 - bitsandbytes.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
Loading ops module
ops module loaded
Registering XPU implementations
Successfully registered XPU implementation
🧪 Running minimal XPU backend test...
int8_linear_matmul_xpu called with tensors of shape: torch.Size([32, 64]) torch.Size([64, 128])

✅ Operator executed successfully!
   Input shapes: torch.Size([32, 64]) x torch.Size([64, 128])
   Output shape: torch.Size([32, 128])
   Output device: xpu:0
[W318 10:31:01.953192430 OperatorEntry.cpp:154] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /pytorch/build/aten/src/ATen/RegisterCPU.cpp:30477
       new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:468 (function operator())
root@pvc-hf-1100-00:/workspace# Initializing bnb_intel module
bnb_intel module initialization complete, _autoload = <function _autoload at 0x7ff6465cea70>
2025-03-18 10:31:03,481 - bitsandbytes.cextension - WARNING - The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization are unavailable.
Loading ops module
ops module loaded
Registering XPU implementations
Successfully registered XPU implementation
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
