# Migration Guide: Python 3.8 to Python 3.12

This guide documents the migration of the FSL_ECG_QA repository from Python 3.8 to Python 3.12.

## Overview

The FSL_ECG_QA project has been migrated to Python 3.12 to take advantage of improved performance, better error messages, and continued support. This migration also includes updating the fairseq-signals dependency to a Python 3.12-compatible fork.

## Key Changes

### 1. Python Version
- **Before:** Python 3.8
- **After:** Python 3.12

### 2. fairseq-signals Dependency
- **Before:** `https://github.com/Jwoo5/fairseq-signals.git` (Python 3.8 only)
- **After:** `https://github.com/sajus2023/fairseq-signals.git` (Python 3.12 compatible fork)

### 3. New Files
- `requirements.txt` - Consolidated list of all Python dependencies
- `MIGRATION.md` - This migration guide

## Installation Instructions

### Prerequisites
Ensure you have Python 3.12 installed on your system:
```bash
python3 --version  # Should output Python 3.12.x
```

### Step 1: Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv_py312
source venv_py312/bin/activate  # On Linux/Mac
# OR
venv_py312\Scripts\activate  # On Windows
```

### Step 2: Install fairseq-signals (Python 3.12 Fork)
```bash
# Clone the Python 3.12 compatible fork
git clone https://github.com/sajus2023/fairseq-signals.git
cd fairseq-signals

# Install in editable mode
pip install --editable ./

# Return to FSL_ECG_QA directory
cd ..
```

### Step 3: Install Project Dependencies
```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Test core imports
python3 -c "
import torch
import numpy as np
import transformers
from omegaconf import OmegaConf
from fairseq_signals import tasks
print('All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
"
```

## Compatibility Notes

### Code Changes
The codebase is largely compatible with Python 3.12 with minimal changes:

1. **extract_model.py:** Added documentation and fixed function signature spacing
2. **All other files:** No changes required - Python 3.8 code is forward-compatible with 3.12

### Dependencies
All dependencies have been updated to versions compatible with Python 3.12:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.24.0
- And others (see [requirements.txt](requirements.txt))

## Known Issues and Solutions

### Issue 1: CUDA Compatibility
**Problem:** PyTorch CUDA version mismatch
**Solution:** Install PyTorch with the correct CUDA version for your system:
```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: fairseq-signals Import Errors
**Problem:** `ModuleNotFoundError: No module named 'fairseq_signals'`
**Solution:** Ensure you've installed the Python 3.12 compatible fork:
```bash
git clone https://github.com/sajus2023/fairseq-signals.git
cd fairseq-signals
pip install --editable ./
```

### Issue 3: NLTK Data Missing
**Problem:** NLTK resources not found when computing BLEU scores
**Solution:** Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Testing the Migration

### Basic Import Test
```bash
python3 -c "
from meta_learner import MetaLearner
from data_loader import FSL_ECG_QA_DataLoader
from meta_trainer import MetaTrainer
print('Core modules imported successfully!')
"
```

### Run a Simple Training Example
Follow the instructions in the main [README.md](README.md) to run training or inference.

## Performance Improvements in Python 3.12

Python 3.12 provides several improvements over 3.8:
- **20-25% faster** execution for many workloads
- **Better error messages** with more context
- **Improved memory efficiency**
- **Better asyncio performance** (if you extend the code)
- **Type hinting improvements** for better IDE support

## Rollback Instructions

If you need to roll back to Python 3.8:
1. Use the original fairseq-signals repository
2. Create a Python 3.8 virtual environment
3. Install dependencies with the versions from the original project

## Support

For issues related to:
- **This migration:** Open an issue in this repository
- **fairseq-signals fork:** Visit https://github.com/sajus2023/fairseq-signals/issues
- **Original FSL_ECG_QA:** Visit https://github.com/Tang-Jia-Lu/FSL_ECG_QA/issues

## Migration Checklist

- [x] Update Python version requirement to 3.12
- [x] Update fairseq-signals dependency to Python 3.12 fork
- [x] Create requirements.txt with all dependencies
- [x] Update README.md with new installation instructions
- [x] Test core imports
- [x] Document migration process
- [ ] (Optional) Run full training/inference test with your data
- [ ] (Optional) Update CI/CD pipelines if applicable

## Contributing

If you find issues with this migration or have improvements to suggest, please:
1. Open an issue describing the problem
2. Submit a pull request with fixes or enhancements
3. Update this migration guide as needed

---

**Last Updated:** December 2025
**Python Version:** 3.12.11
**Migrated By:** sajus2023
