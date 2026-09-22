"""Quick environment check script (no heavy model loading)."""
import sys
print("=" * 56)
print("Python:", sys.version.split()[0], "| Platform:", sys.platform)
print("=" * 56)

errors = []

print("\n[1/3] Core pipeline import (LangGraph + schemas)...")
try:
    from agent.graph import build_graph
    g = build_graph()
    print("  ✅ build_graph() ->", type(g).__name__)
except Exception as e:
    print("  ❌ FAIL:", e)
    errors.append(("graph import", e))

print("\n[2/3] Lightweight dependencies...")
for mod, pip_name in [
    ("yaml", "PyYAML"),
    ("pydantic", "pydantic"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("httpx", "httpx"),
    ("tqdm", "tqdm"),
]:
    try:
        __import__(mod)
        print(f"  ✅ {pip_name}")
    except Exception as e:
        print(f"  ❌ {pip_name}: {type(e).__name__}")
        errors.append((pip_name, e))

print("\n[3/3] Heavy Apple Silicon deps (mlx / mlx-lm / mflux / langgraph)...")
for mod, pip_name in [
    ("langgraph", "langgraph"),
    ("langchain", "langchain"),
    ("mlx", "mlx"),
    ("mlx_lm", "mlx-lm"),
    ("mflux", "mflux"),
]:
    try:
        __import__(mod)
        print(f"  ✅ {pip_name}")
    except Exception as e:
        print(f"  ⚠️  {pip_name}: {type(e).__name__}")
        errors.append((pip_name, e))

print()
print("=" * 56)
if errors:
    print(f"RESULT: {len(errors)} issue(s). See above.")
    print("Install missing:  pip install -r requirements.txt")
else:
    print("🎉 All checks passed! Ready to run main.py")
print("=" * 56)
sys.exit(0 if not errors else 1)
