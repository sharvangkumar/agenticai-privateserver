"""
quantization_demo.py — What is quantization? A hands-on demo.

Run this FIRST to understand what quantization means before pulling models.
No Ollama needed — pure Python math demo.
"""

import struct
import sys

# ══════════════════════════════════════════════════════════════════
# WHAT IS QUANTIZATION? (the concept in plain code)
#
# An LLM is millions of numbers (weights). By default they are
# stored as float32 (4 bytes each). Quantization shrinks them
# to int8 or int4 (1–2 bytes), saving RAM at a small quality cost.
# ══════════════════════════════════════════════════════════════════

def bytes_of(value, dtype="float32") -> int:
    sizes = {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}
    return sizes[dtype]

def model_ram_gb(num_parameters_billions: float, dtype: str) -> float:
    """Estimate RAM needed for a model."""
    params = num_parameters_billions * 1_000_000_000
    return round(params * bytes_of(None, dtype) / 1e9, 2)

def quantize_weight(value: float, bits: int = 8) -> tuple:
    """
    Simulate quantizing one float weight to N-bit integer.
    Returns (quantized_int, dequantized_float, error).
    """
    levels = 2 ** bits
    # scale to [0, levels-1]
    min_val, max_val = -3.0, 3.0    # typical weight range
    scale = (max_val - min_val) / (levels - 1)
    quantized = round((value - min_val) / scale)
    quantized = max(0, min(levels - 1, quantized))
    # dequantize back
    dequantized = quantized * scale + min_val
    error = abs(value - dequantized)
    return quantized, dequantized, error


def demo_single_weight():
    print("\n" + "═"*55)
    print("  DEMO 1: Quantizing a single weight")
    print("═"*55)
    original = 1.2345678901234567   # a float32 weight

    print(f"\nOriginal float32 weight : {original:.10f}")
    print(f"Memory used             : {struct.calcsize('f')} bytes\n")

    for bits in (16, 8, 4):
        q, dq, err = quantize_weight(original, bits)
        mem = bytes_of(None, f"int{bits}" if bits <= 8 else "float16")
        label = f"q{bits}"
        print(f"  {label:4s}  stored={q:5d}  recovered={dq:.6f}  "
              f"error={err:.6f}  mem={mem}B")


def demo_model_sizes():
    print("\n" + "═"*55)
    print("  DEMO 2: RAM needed for phi4-mini (3.8B params)")
    print("═"*55)
    params = 3.8   # billion

    rows = [
        ("fp32",    "float32", "full precision"),
        ("fp16",    "float16", "half precision"),
        ("q8",      "int8",    "8-bit quant"),
        ("q4",      "int4",    "4-bit quant  ← YOU WANT THIS"),
    ]
    print(f"\n  {'Format':<8} {'RAM (GB)':<12} {'Notes'}")
    print(f"  {'-'*45}")
    for name, dtype, note in rows:
        gb = model_ram_gb(params, dtype)
        marker = " ✓" if "YOU WANT" in note else ""
        print(f"  {name:<8} {gb:<12} {note}{marker}")

    print(f"\n  Your laptop: 16 GB RAM")
    print(f"  OS + browser uses ~4 GB → ~12 GB free for model")
    print(f"  q4_K_M for phi4-mini uses ~2.5 GB  →  safe and fast\n")


def demo_quality_vs_size():
    print("\n" + "═"*55)
    print("  DEMO 3: Quality vs size trade-off")
    print("═"*55)

    # Simulated benchmark scores (approximate, based on public evals)
    models = [
        ("phi4-mini  fp16",   7.6,  95),
        ("phi4-mini  q8_0",   3.9,  94),
        ("phi4-mini  q4_K_M", 2.5,  92),
        ("phi4-mini  q2_K",   1.4,  80),
    ]
    print(f"\n  {'Model':<22} {'RAM(GB)':<10} {'Quality%'}")
    print(f"  {'-'*42}")
    for name, ram, quality in models:
        bar = "█" * (quality // 10)
        print(f"  {name:<22} {ram:<10} {quality}%  {bar}")

    print("\n  q4_K_M gives 92% quality at 33% of the RAM. Best choice.")


def how_to_pull():
    print("\n" + "═"*55)
    print("  HOW TO PULL THE RIGHT QUANTIZED MODEL")
    print("═"*55)
    print("""
  # Best choice for your 16GB laptop (no GPU):
  ollama pull phi4-mini                    ← Ollama picks q4 automatically

  # To pick explicitly:
  ollama pull phi4-mini:3.8b-instruct-q4_K_M   ← best balance
  ollama pull phi4-mini:3.8b-instruct-q8_0      ← better quality, 3.9 GB

  # For coding:
  ollama pull qwen2.5-coder:3b             ← ~2 GB, fast autocomplete

  # Check what's installed:
  ollama list

  # The 'K_M' in q4_K_M means:
  #   K = uses k-means clustering for better quality
  #   M = medium — balances speed vs quality within k-quants
  """)


if __name__ == "__main__":
    demo_single_weight()
    demo_model_sizes()
    demo_quality_vs_size()
    how_to_pull()
