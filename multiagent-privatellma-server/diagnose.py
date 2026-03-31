"""
diagnose.py — Run this before agents.py to check everything is working.
Usage: python diagnose.py
"""
import sys
import requests
from openai import OpenAI

OLLAMA_URL = "http://localhost:11434"

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("\n" + "="*55)
print("  Ollama + LLM Diagnostic")
print("="*55)

# 1. Is Ollama running at all?
ok1 = check(
    "Ollama reachable",
    lambda: requests.get(f"{OLLAMA_URL}/", timeout=3).status_code
)

# 2. List models
def list_models():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
    names = [m["name"] for m in r.get("models", [])]
    if not names:
        raise Exception("No models installed! Run: ollama pull phi4-mini")
    return names

ok2 = check("Models installed", list_models)

# 3. Can we call the OpenAI-compatible endpoint?
def test_openai_sdk():
    client = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
    r = client.chat.completions.create(
        model="phi4-mini",
        messages=[{"role": "user", "content": "say hi in 3 words"}],
        max_tokens=20,
        timeout=30,
    )
    return r.choices[0].message.content.strip()

ok3 = check("OpenAI SDK call (phi4-mini)", test_openai_sdk)

print()
if not ok1:
    print("  FIX: Open a NEW terminal and run:  ollama serve")
    print("       Keep that terminal open, then re-run this script.")
elif not ok2:
    print("  FIX: Run:  ollama pull phi4-mini")
elif not ok3:
    print("  FIX: Model may still be loading. Wait 10 seconds and retry.")
    print("       Or check: ollama ps   (to see what's running)")
else:
    print("  All checks passed. You can now run:  python agents.py")

print()
