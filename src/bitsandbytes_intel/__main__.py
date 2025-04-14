import sys

import torch

import bitsandbytes  # noqa


def main():
    print("🧪 Running minimal XPU backend test...")

    try:
        A = torch.randint(-128, 127, (32, 64), dtype=torch.int8).to("xpu")
        B = torch.randint(-128, 127, (128, 64), dtype=torch.int8).to("xpu")

        result = torch.ops.bitsandbytes.int8_linear_matmul(A, B)
        # Simple output verification
        print("\n✅ Operator executed successfully!")
        print(f"   Input shapes: {A.shape} x {B.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Output device: {result.device}")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Test failed: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
