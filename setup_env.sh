#!/bin/bash
set -e  # stop on any error
ENV_NAME="slm_env"
PYTHON_VERSION="python3"  # change to your python version

echo "================================================"
echo " LLM Eval Harness — Local Setup (RTX 3050)"
echo "================================================"

# --- 1. Create virtual env ---

echo -e "\e[1;36m[1/5]\e[0m \e[32mCreating virtual environment:\e[0m \e[1m$ENV_NAME\e[0m"
$PYTHON_VERSION -m venv $ENV_NAME

# --- 2. Activate it ---

echo -e "\e[1;36m[2/5]\e[0m \e[32mActivating environment...\e[0m"
source $ENV_NAME/bin/activate


echo "Checking Python version..."
python3 -c "
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    print(f'ERROR: Python 3.10+ required, found {major}.{minor}')
    sys.exit(1)
print(f'  Python {major}.{minor} — OK')
"

# --- 3. Upgrade pip ---
echo -e "\e[1;36m[3/5]\e[0m \e[32mUpgrading pip...\e[0m"
pip install --upgrade pip wheel setuptools

# --- 4. Install PyTorch with CUDA 11.8 (compatible with RTX 3050) ---
echo -e "\e[1;36m[4/5]\e[0m \e[32mInstalling \e[1mPyTorch\e[0m with \e[1;33mCUDA 11.8\e[0m..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# --- 5. Install project dependencies ---
echo -e "\e[1;36m[5/5]\e[0m \e[32mInstalling project dependencies...\e[0m"
pip install \
    lm-eval \
    transformers \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    einops \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    fpdf2 \
    jinja2 \
    streamlit \
    pyyaml \
    tqdm \
    huggingface_hub \
    scipy \
    datasets \
    pytest \
    packaging \
    rouge-score \
    nltk

# --- Verify CUDA is available ---
echo ""
echo "Verifying CUDA availability..."
python3 -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU detected    : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA not detected — will run on CPU (slow)')
"

echo ""
echo "================================================"
echo " Setup complete! Activate with:"
echo "   source $ENV_NAME/bin/activate"
echo "================================================"