#!/usr/bin/env python3
"""
Installation Verification Script for FSL_ECG_QA (Python 3.12)

This script checks if all dependencies are installed correctly
and if the environment is properly configured.
"""

import sys
import importlib

def check_python_version():
    """Check if Python version is 3.12 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.12+ required, but found {version.major}.{version.minor}.{version.micro}")
        return False

def check_module(module_name, package_name=None):
    """Check if a module can be imported and return its version."""
    display_name = package_name if package_name else module_name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: Not installed ({e})")
        return False

def check_cuda():
    """Check CUDA availability for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available (CPU only)")
            return True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_project_modules():
    """Check if project-specific modules can be imported."""
    print("\nChecking project modules...")
    modules = [
        'meta_learner',
        'meta_trainer',
        'data_loader',
        'extract_model',
        'load_class',
        'compute_scores',
        'utils'
    ]

    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_ok = False

    return all_ok

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("FSL_ECG_QA Installation Verification")
    print("=" * 60)
    print()

    results = []

    # Check Python version
    results.append(check_python_version())
    print()

    # Check core dependencies
    print("Checking core dependencies...")
    results.append(check_module('torch', 'PyTorch'))
    results.append(check_module('torchvision'))
    results.append(check_module('torchaudio'))
    results.append(check_module('numpy', 'NumPy'))
    results.append(check_module('scipy', 'SciPy'))
    print()

    # Check ML/NLP libraries
    print("Checking ML/NLP libraries...")
    results.append(check_module('transformers'))
    results.append(check_module('evaluate'))
    results.append(check_module('nltk', 'NLTK'))
    print()

    # Check configuration libraries
    print("Checking configuration libraries...")
    results.append(check_module('hydra', 'Hydra'))
    results.append(check_module('omegaconf', 'OmegaConf'))
    print()

    # Check fairseq-signals
    print("Checking fairseq-signals...")
    results.append(check_module('fairseq_signals'))
    print()

    # Check utility libraries
    print("Checking utility libraries...")
    results.append(check_module('sklearn', 'scikit-learn'))
    results.append(check_module('wfdb', 'WFDB'))
    results.append(check_module('tqdm'))
    results.append(check_module('pandas'))
    results.append(check_module('bert_score', 'bert-score'))
    results.append(check_module('rouge_score', 'rouge-score'))
    print()

    # Check CUDA
    print("Checking CUDA availability...")
    check_cuda()
    print()

    # Check project modules
    project_modules_ok = check_project_modules()
    results.append(project_modules_ok)
    print()

    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)

    if all(results):
        print("✓ All checks passed!")
        print()
        print("Your environment is ready to use FSL_ECG_QA.")
        print()
        print("Next steps:")
        print("1. Download the ECG-QA dataset")
        print("2. Download pre-trained model weights")
        print("3. Update data paths in train.py and inference.py")
        print("4. Run training or inference")
        return 0
    else:
        print("✗ Some checks failed!")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("For fairseq-signals, see MIGRATION.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
