import numpy as np
import torch
import torchvision.models as models


def inspect_weights(model: torch.nn.Module) -> None:
    layer = model.layer1[0].conv1
    weights = layer.weight.data.numpy()

    print("=" * 50)
    print("STEP 1: Raw FP32 weights")
    print("=" * 50)
    print(f"  dtype:  {weights.dtype}")
    print(f"  shape:  {weights.shape}")
    print(f"  range:  {weights.min():.6f}  to  {weights.max():.6f}")
    print(f"  size in memory: {weights.nbytes / 1024:.1f} KB")
    return weights

def manual_quantize(weights: np.ndarray) -> tuple:
    print("\n" + "=" * 50)
    print("STEP 2: Manual INT8 quantization")
    print("=" * 50)

    scale = max(abs(weights.max()), abs(weights.min())) / 127.0
    print(f"  scale factor: {scale:.8f}")

    weights_int8 = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)

    print(f"  INT8 dtype:   {weights_int8.dtype}")
    print(f"  INT8 range:   {weights_int8.min()}  to  {weights_int8.max()}")
    print(f"  size in memory: {weights_int8.nbytes / 1024:.1f} KB  "
          f"({weights.nbytes / weights_int8.nbytes}x smaller)")

    return weights_int8, scale

if __name__ == "__main__":

    model = models.resnet50(weights="DEFAULT").eval()
    weights_fp32 = inspect_weights(model)
    weights_int8, scale = manual_quantize(weights_fp32)



