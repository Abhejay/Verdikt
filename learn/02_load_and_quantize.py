from __future__ import annotations

import torch
import numpy as np
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = "The capital of France is"

def load_bf16(model_id: str) -> tuple:
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN
    )

    model.eval()

    elapsed = time.time() - start
    param_count = sum(p.numel() for p in model.parameters())
    memory_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1024 / 1024

    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Parameters: {param_count / 1e6:.1f}M")
    print(f"  Memory footprint: {memory_mb:.0f} MB")
 
    return model, tokenizer

def load_int8(model_id: str) -> tuple:
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        token=HF_TOKEN
    )



    

