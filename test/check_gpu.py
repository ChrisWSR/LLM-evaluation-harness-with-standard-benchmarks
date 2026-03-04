"""
Run this first to diagnose the onnxruntime-gpu issue.
python check_ort_gpu.py
"""

import subprocess
import sys

print("=" * 50)
print("OnnxRuntime GPU Diagnostic")
print("=" * 50)

# 1. Check which onnxruntime is installed
print("\n1. Installed onnxruntime packages:")
result = subprocess.run(
    ["pip", "show", "onnxruntime", "onnxruntime-gpu"],
    capture_output=True, text=True
)
print(result.stdout or "  ❌ Neither found")

# 2. Check ORT providers
print("\n2. Available ORT providers:")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    for p in providers:
        tag = "✅" if "CUDA" in p else "  "
        print(f"  {tag} {p}")
    if "CUDAExecutionProvider" not in providers:
        print("\n  ❌ CUDAExecutionProvider missing — this is the problem")
except Exception as e:
    print(f"  ❌ Error importing onnxruntime: {e}")

# 3. Check CUDA
print("\n3. CUDA / PyTorch:")
try:
    import torch
    print(f"  torch version  : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version   : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ❌ {e}")

# 4. Check cuDNN
print("\n4. cuDNN:")
try:
    import torch
    print(f"  cuDNN version  : {torch.backends.cudnn.version()}")
    print(f"  cuDNN enabled  : {torch.backends.cudnn.enabled}")
except Exception as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 50)
print("RECOMMENDED FIX based on results above:")
print("=" * 50)

try:
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        import torch
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
        if cuda_ver:
            major = cuda_ver.split(".")[0]
            print(f"\n  Your CUDA version: {cuda_ver}")
            print(f"\n  Run this:")
            print(f"\n  pip uninstall onnxruntime onnxruntime-gpu -y")
            if major == "12":
                print(f"  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/")
            else:
                print(f"  pip install onnxruntime-gpu")
        else:
            print("\n  ❌ CUDA not available in PyTorch — reinstall PyTorch with CUDA first")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
except:
    pass