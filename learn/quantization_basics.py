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

def measure_loss(weights_fp32: np.ndarray, weights_int8: np.ndarray, scale: float) -> None:
    print("\n" + "=" * 50)
    print("STEP 3: Dequantize and measure error")
    print("=" * 50)

    weights_reconstructed = weights_int8.astype(np.float32) * scale
    error = np.abs(weights_fp32 - weights_reconstructed)

    weight_range = weights_fp32.max() - weights_fp32.min()
    
    print(f"  MAE:                {np.mean(error):.8f}")
    print(f"  Max error:          {np.max(error):.8f}")
    print(f"  % of weight range:  {np.mean(error) / weight_range * 100:.4f}%")
    print(f"  Std of error:       {np.std(error):.8f}")

    print("\n  What this means:")
    print(f"  Every weight in this layer shifts by ~{np.mean(error):.6f} on average.")
    print(f"  Across all layers in ResNet-50, these errors compound.")
    print(f"  Verdikt measures the cumulative effect at the model output.")

def compare_forward_pass(model: torch.nn.Module) -> None:
    print("\n" + "=" * 50)
    print("STEP 4: Output divergence — FP32 vs quantized forward pass")
    print("=" * 50)

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out_fp32 = model(x).numpy()

    model_int8 = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    with torch.no_grad():
        out_int8 = model_int8(x).numpy()

    delta = np.abs(out_fp32 - out_int8)
    print(f"  Output shape:     {out_fp32.shape}")
    print(f"  FP32 range:       {out_fp32.min():.4f}  to  {out_fp32.max():.4f}")
    print(f"  MAE (output):     {np.mean(delta):.6f}")
    print(f"  Max delta:        {np.max(delta):.6f}")

    fp32_class = np.argmax(out_fp32)
    int8_class = np.argmax(out_int8)
    print(f"\n  FP32 predicted class: {fp32_class}")
    print(f"  INT8 predicted class: {int8_class}")
    print(f"  Same prediction:      {fp32_class == int8_class}")
    print("\n  This is the core question Verdikt answers at scale:")
    print("  Is this delta bounded enough to trust in a safety-critical system?")

if __name__ == "__main__":

    model = models.resnet50(weights="DEFAULT").eval()
    weights_fp32 = inspect_weights(model)
    weights_int8, scale = manual_quantize(weights_fp32)

    measure_loss(weights_fp32, weights_int8, scale)
    compare_forward_pass(model)


