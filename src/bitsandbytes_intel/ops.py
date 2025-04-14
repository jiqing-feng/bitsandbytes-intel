import torch

print("Loading ops module")


def register_ops():
    print("Registering XPU implementations")

    # Check if the operator exists
    if not hasattr(torch.ops.bitsandbytes, "int8_linear_matmul"):
        raise RuntimeError("bitsandbytes::int8_linear_matmul not found! Make sure bitsandbytes is installed")

    @torch.library.impl("bitsandbytes::int8_linear_matmul", "XPU")
    def int8_linear_matmul_xpu(A: torch.Tensor, B: torch.Tensor):
        print("int8_linear_matmul_xpu called with tensors of shape:", A.shape, B.shape)
        return torch.matmul(A, B)

    print("Successfully registered XPU implementation")


print("ops module loaded")
