"""
loader.py — Model Loader
========================
Handles loading HuggingFace models onto GPU with optional 4-bit quantization
via bitsandbytes. Also provides clean unload to free VRAM between runs.

Supports:
  - Full precision (fp16) for models < 4B params
  - 4-bit quantization (NF4) for 7B models on 8GB VRAM (RTX 3050)
  - Auto device mapping for multi-GPU (future-proof)
"""

import gc
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def build_bnb_config() -> BitsAndBytesConfig:
    """
    4-bit NF4 quantization config for bitsandbytes.
    Reduces a 7B model from ~14GB → ~5GB VRAM — fits a 3050 8GB.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 — best accuracy at 4-bit
        bnb_4bit_use_double_quant=True,      # nested quantization → extra ~0.4 bits saved
        bnb_4bit_compute_dtype=torch.float16,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_cfg: dict) -> tuple:
    """
    Load a model and tokenizer from HuggingFace Hub.

    Args:
        model_cfg: dict from configs/models.yaml, e.g.:
          {
            name:         "mistral-7b",
            hf_path:      "mistralai/Mistral-7B-Instruct-v0.2",
            device:       "cuda",
            load_in_4bit: true
          }

    Returns:
        (model, tokenizer) tuple — both ready for inference.
    """
    hf_path     = model_cfg["hf_path"]
    device      = model_cfg.get("device", "cuda")
    use_4bit    = model_cfg.get("load_in_4bit", False)

    # Fallback to CPU if CUDA isn't available
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"

    log.info(f"Loading tokenizer: {hf_path}")
    tokenizer = _load_tokenizer(hf_path)

    log.info(f"Loading model: {hf_path}  |  4-bit={use_4bit}  |  device={device}")
    model = _load_model(hf_path, device, use_4bit)

    _log_vram_usage(hf_path)

    return model, tokenizer


def _load_tokenizer(hf_path: str):
    """Load tokenizer with safe defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        hf_path,
        trust_remote_code=True,
        use_fast=True,
    )
    # Some models (e.g. LLaMA family) have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.debug("pad_token set to eos_token.")
    return tokenizer


def _load_model(hf_path: str, device: str, use_4bit: bool):
    """Load the causal LM with the appropriate precision."""
    common_kwargs = dict(
        pretrained_model_name_or_path=hf_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if use_4bit:
        # 4-bit quantized — device_map handles placement automatically
        model = AutoModelForCausalLM.from_pretrained(
            **common_kwargs,
            quantization_config=build_bnb_config(),
            device_map="auto",
        )
    elif device == "cuda":
        # fp16 on GPU — cleanest scores for benchmarking small models
        model = AutoModelForCausalLM.from_pretrained(
            **common_kwargs,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # CPU fallback — fp32
        model = AutoModelForCausalLM.from_pretrained(
            **common_kwargs,
            torch_dtype=torch.float32,
        )

    model.eval()  # disable dropout, etc.
    return model


# ══════════════════════════════════════════════════════════════════════════════
# UNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def unload_model(model) -> None:
    """
    Delete model from memory and flush GPU cache.
    Always called in main.py's finally block so VRAM
    is freed before loading the next model.
    """
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log.info("Model unloaded and VRAM flushed.")
    except Exception as e:
        log.warning(f"Unload warning (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _log_vram_usage(label: str = "") -> None:
    """Log current GPU memory usage after loading."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated()  / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"VRAM after loading {label}: {used:.2f} GB / {total:.1f} GB")


def get_device_info() -> dict:
    """Return a summary dict of the current GPU state."""
    if not torch.cuda.is_available():
        return {"cuda": False, "device": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda"       : True,
        "device_name": props.name,
        "vram_total" : round(props.total_memory / 1e9, 1),
        "vram_used"  : round(torch.cuda.memory_allocated() / 1e9, 2),
    }