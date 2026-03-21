# Verdikt

**Verifiable safety evidence for quantized neural network deployment in regulated systems.**

Verdikt is an open-source toolkit that systematically generates structured safety evidence for ONNX/TensorRT quantized models. It answers a question existing tooling ignores: not just whether a quantized model is accurate, but whether its behavior under quantization is **bounded, reproducible, and verifiable** in a form a safety auditor can accept.

> 📄 Paper: *"Verdikt: Towards Verifiable Safety Evidence for Quantized Neural Network Deployment in Regulated Systems"* — NeurIPS 2026 Workshop (TrustworthyML / SafeAI) — *forthcoming*

---

## What Verdikt produces

| Module | Evidence artifact |
|---|---|
| `divergence` | Per-layer and output-level delta across FP32 / FP16 / INT8 with configurable safety thresholds |
| `nondeterminism` | Output variance statistics across N inference runs under thermal and load variation |
| `attest` | Cryptographically signed build manifest — TRT version, CUDA version, calibration hash, engine fingerprint |
| `gap_report` | Structured mapping of measured properties against ISO/PAS 8800 evidence requirements |

---

## Quickstart

```bash
git clone https://github.com/Abhejay/verdikt.git
cd verdikt
pip install -e ".[dev]"

# Export a model to ONNX
python scripts/export_model.py --model resnet50 --output experiments/models/

# Run full evidence pipeline
python scripts/run_pipeline.py \
    --model experiments/models/resnet50.onnx \
    --calibration experiments/calibration/ \
    --output experiments/results/
```

---

## Experimental setup

All results in the paper are produced on:

- **Platform**: GCP `n1-standard-4` + NVIDIA T4 (16GB)
- **TensorRT**: 10.x (pinned — see `docker/Dockerfile`)
- **CUDA**: 12.x
- **JetPack / Driver**: see `docker/Dockerfile`
- **Python**: 3.10

Every experiment is fully reproducible via the provided Docker container.

---

## Model suite

Eight architectures selected for regulated deployment relevance:

| Model | Task | Domain relevance |
|---|---|---|
| ResNet-50 | Classification | Baseline reference |
| MobileNetV3-Large | Classification | Edge-optimized baseline |
| EfficientNet-B0 | Classification | Embedded baseline |
| YOLOv8n | Object detection | ADAS, industrial |
| YOLO-NAS-S | Object detection | Recent architecture |
| EfficientDet-D0 | Object detection | Medical imaging |
| DepthAnything-S | Depth estimation | ADAS perception |
| PIDNet-S | Semantic segmentation | Autonomous systems |

---

## Standards coverage

Verdikt's gap report maps measured evidence against:

- **ISO/PAS 8800** — AI safety for road vehicles
- **IEC 62304** — Medical device software lifecycle

---

## Project structure

```
verdikt/
├── verdikt/
│   ├── core/
│   │   ├── divergence.py       # Precision divergence measurement
│   │   ├── nondeterminism.py   # Variance profiling across N runs
│   │   ├── attest.py           # Signed build manifest generation
│   │   └── gap_report.py       # ISO/PAS 8800 evidence gap mapping
│   ├── agent/
│   │   ├── validator_agent.py  # Agentic orchestration layer (MLSys extension)
│   │   └── tools.py            # Agent-callable tool wrappers
│   └── utils/
│       ├── export.py           # PyTorch → ONNX export utilities
│       ├── calibration.py      # INT8 calibration dataset helpers
│       └── logging.py          # Structured result logging
├── experiments/
│   ├── models/                 # ONNX model files
│   ├── calibration/            # INT8 calibration datasets
│   └── results/                # JSON evidence artifacts
├── scripts/
│   ├── export_model.py         # Model export entrypoint
│   ├── run_pipeline.py         # Full evidence pipeline entrypoint
│   └── setup_gcp.sh            # GCP instance setup script
├── tests/
├── docs/
│   └── evidence_schema.md      # Evidence artifact schema documentation
├── docker/
│   └── Dockerfile              # Pinned TRT + CUDA environment
└── pyproject.toml
```

---

## Reproducing paper results

```bash
# Pull the pinned Docker environment
docker pull ghcr.io/abhejay/verdikt:latest

# Run the full experiment suite
docker run --gpus all verdikt:latest \
    python scripts/run_pipeline.py --all-models
```

All results are written to `experiments/results/` as structured JSON evidence artifacts.

---

## Citation

```bibtex
@article{murali2026verdikt,
  title={Verdikt: Towards Verifiable Safety Evidence for Quantized Neural Network Deployment in Regulated Systems},
  author={Murali, Abhejay},
  journal={NeurIPS Workshop on Trustworthy Machine Learning},
  year={2026}
}
```

---

## License

Apache 2.0 — see `LICENSE`.
