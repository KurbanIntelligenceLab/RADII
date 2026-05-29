#!/bin/bash
# =============================================================================
# Vast.ai setup and launch script for published-scale training
#
# Usage (on the Vast.ai instance):
#   bash scripts/vastai_setup.sh
#
# Prerequisites:
#   - rsync or git clone the RADII repo to ~/RADII
#   - rsync the radii/ dataset (or at least radii/cache/ + radii/unit_cells/)
#
# Data transfer examples (run from your local machine):
#   # Option A: full dataset
#   rsync -avz --progress /Users/jp/RADII/ vast_instance:~/RADII/
#
#   # Option B: cache only (faster, ~few hundred MB)
#   rsync -avz /Users/jp/RADII/radii/cache/ vast_instance:~/RADII/radii/cache/
#   rsync -avz /Users/jp/RADII/radii/unit_cells/ vast_instance:~/RADII/radii/unit_cells/
# =============================================================================
set -e

REPO_DIR="${HOME}/RADII"
cd "${REPO_DIR}"

echo "================================================"
echo "  Published-Scale Training Setup (Vast.ai)"
echo "================================================"

# -----------------------------------------------------------------
# 1. Environment setup
# -----------------------------------------------------------------
echo "[1/4] Setting up Python environment..."

# Check if conda is available; if not, install miniconda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    source ~/.bashrc
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "radii"; then
    echo "Creating radii conda environment..."
    conda create -n radii python=3.10 -y
fi

# Activate and install dependencies
eval "$(conda shell.bash hook)"
conda activate radii

echo "Installing PyTorch + CUDA..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing PyG and extensions..."
pip install -q torch-geometric
pip install -q torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

echo "Installing remaining dependencies..."
pip install -q scipy pandas pymatgen tqdm

# -----------------------------------------------------------------
# 2. Verify setup
# -----------------------------------------------------------------
echo ""
echo "[2/4] Verifying setup..."

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

# Verify dataset exists
if [ ! -d "radii" ]; then
    echo "ERROR: radii/ dataset directory not found. Please rsync it first."
    exit 1
fi

# Warm up dataset cache
echo "Building dataset cache (if needed)..."
python -c "from radii.data import RADIIDataloader; RADIIDataloader('radii', loaded_frac=0.5); print('Dataset cache ready.')"

# -----------------------------------------------------------------
# 3. Create log directory
# -----------------------------------------------------------------
echo ""
echo "[3/4] Setting up logging..."
mkdir -p logs

# -----------------------------------------------------------------
# 4. Launch training (one model per GPU)
# -----------------------------------------------------------------
echo ""
echo "[4/4] Launching training..."

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

if [ "${GPU_COUNT}" -ge 2 ]; then
    echo "Detected ${GPU_COUNT} GPUs. Running DiffCSP on GPU 0, MatterGen on GPU 1."

    CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.train_published_scale \
        --model diffcsp --resume \
        > logs/diffcsp_published.log 2>&1 &
    DIFFCSP_PID=$!

    CUDA_VISIBLE_DEVICES=1 nohup python -m scripts.train_published_scale \
        --model mattergen --resume \
        > logs/mattergen_published.log 2>&1 &
    MATTERGEN_PID=$!

    echo ""
    echo "Training launched!"
    echo "  DiffCSP PID: ${DIFFCSP_PID} (GPU 0)"
    echo "  MatterGen PID: ${MATTERGEN_PID} (GPU 1)"
else
    echo "Detected ${GPU_COUNT} GPU(s). Running models sequentially."

    CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.train_published_scale \
        --model diffcsp --resume \
        > logs/diffcsp_published.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.train_published_scale \
        --model mattergen --resume \
        > logs/mattergen_published.log 2>&1 &

    echo "Training launched sequentially on GPU 0."
fi

echo ""
echo "========================================"
echo "  Monitoring commands:"
echo "========================================"
echo "  tail -f logs/diffcsp_published.log"
echo "  tail -f logs/mattergen_published.log"
echo ""
echo "  Check seed completion:"
echo "  ls results/task_1/*_published/SEED_*_DONE"
echo ""
echo "  Download seed 1 results (from local machine):"
echo "  scp -r vast:~/RADII/results/task_1/diffcsp_published/1/ results/task_1/diffcsp_published/1/"
echo "  scp -r vast:~/RADII/results/task_1/mattergen_published/1/ results/task_1/mattergen_published/1/"
echo "========================================"
