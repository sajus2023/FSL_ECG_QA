#!/bin/bash

# FSL_ECG_QA Setup Script for Python 3.12
# This script automates the installation process

set -e  # Exit on error

echo "========================================="
echo "FSL_ECG_QA Python 3.12 Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -ne 3 ] || [ "$MINOR_VERSION" -lt 12 ]; then
    echo "Error: Python 3.12 or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Check if virtual environment should be created
read -p "Create a new virtual environment? (recommended) [y/N]: " CREATE_VENV
if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment 'venv_fsl_ecg'..."
    python3 -m venv venv_fsl_ecg
    echo "✓ Virtual environment created"
    echo ""
    echo "Activating virtual environment..."
    source venv_fsl_ecg/bin/activate
    echo "✓ Virtual environment activated"
    echo ""
fi

# Install fairseq-signals
echo "Installing fairseq-signals (Python 3.12 fork)..."
read -p "Have you already cloned fairseq-signals from https://github.com/sajus2023/fairseq-signals? [y/N]: " FAIRSEQ_CLONED

if [[ ! $FAIRSEQ_CLONED =~ ^[Yy]$ ]]; then
    echo "Cloning fairseq-signals..."
    PARENT_DIR=$(dirname $(pwd))
    cd "$PARENT_DIR"

    if [ -d "fairseq-signals" ]; then
        echo "Warning: fairseq-signals directory already exists."
        read -p "Do you want to remove it and re-clone? [y/N]: " REMOVE_EXISTING
        if [[ $REMOVE_EXISTING =~ ^[Yy]$ ]]; then
            rm -rf fairseq-signals
            git clone https://github.com/sajus2023/fairseq-signals.git
        fi
    else
        git clone https://github.com/sajus2023/fairseq-signals.git
    fi

    cd fairseq-signals
    echo "Installing fairseq-signals in editable mode..."
    pip install --editable ./
    cd -
    echo "✓ fairseq-signals installed"
else
    echo "Skipping fairseq-signals clone. Make sure it's installed with 'pip install --editable ./'"
fi
echo ""

# Install PyTorch
echo "Installing PyTorch..."
read -p "Select PyTorch installation: [1] CUDA 11.8, [2] CUDA 12.1, [3] CPU only [1]: " PYTORCH_OPTION
PYTORCH_OPTION=${PYTORCH_OPTION:-1}

case $PYTORCH_OPTION in
    1)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Invalid option. Defaulting to CUDA 11.8"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
esac
echo "✓ PyTorch installed"
echo ""

# Install remaining dependencies
echo "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print('✓ NLTK data downloaded')
except Exception as e:
    print(f'Warning: Could not download NLTK data: {e}')
"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
try:
    import torch
    import numpy as np
    import transformers
    from omegaconf import OmegaConf
    from fairseq_signals import tasks
    print('✓ All core imports successful!')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
    print(f'  NumPy: {np.__version__}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)
"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Update the data paths in train.py and inference.py to match your setup"
echo "2. Download the ECG-QA dataset from: https://huggingface.co/datasets/jialucode/FSL_ECG_QA_Dataset/"
echo "3. Download pre-trained model weights from: https://huggingface.co/jialucode/FSL_ECG_QA/"
echo "4. Run training: python train.py --help"
echo "5. Run inference: python inference.py --help"
echo ""
echo "For more information, see:"
echo "- README.md: General project information"
echo "- MIGRATION.md: Python 3.12 migration details"
echo ""

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    echo "Note: Remember to activate the virtual environment before running scripts:"
    echo "  source venv_fsl_ecg/bin/activate"
    echo ""
fi
