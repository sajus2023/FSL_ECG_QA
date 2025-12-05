# Quick Start Guide - FSL_ECG_QA (Python 3.12)

This guide will get you up and running with FSL_ECG_QA in just a few steps.

## Prerequisites

- Python 3.12 installed on your system
- Git installed
- CUDA-capable GPU (optional but recommended)
- At least 10GB of free disk space

## Installation (5 minutes)

### Option 1: Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/sajus2023/FSL_ECG_QA.git
cd FSL_ECG_QA

# Run the setup script
chmod +x setup.sh
./setup.sh
```

The script will:
1. Check your Python version
2. Optionally create a virtual environment
3. Install fairseq-signals
4. Install all dependencies
5. Download NLTK data
6. Verify the installation

### Option 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/sajus2023/FSL_ECG_QA.git
cd FSL_ECG_QA

# Install fairseq-signals
cd ..
git clone https://github.com/sajus2023/fairseq-signals.git
cd fairseq-signals
pip install --editable ./
cd ../FSL_ECG_QA

# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Verify Installation

Run the verification script:
```bash
python3 verify_installation.py
```

You should see all checks pass with ✓ marks.

## Download Data and Models

### 1. Download ECG-QA Dataset
```bash
# Download from Hugging Face
# Visit: https://huggingface.co/datasets/jialucode/FSL_ECG_QA_Dataset/

# Example structure:
# ecg_qa/
# ├── ptbxl/
# │   └── paraphrased/
# │       ├── train/
# │       ├── valid/
# │       └── test/
# └── mimic/
#     └── paraphrased/
#         ├── train/
#         ├── valid/
#         └── test/
```

### 2. Download Pre-trained Weights
```bash
# Download from Hugging Face
# Visit: https://huggingface.co/jialucode/FSL_ECG_QA/

# Save to: ./ecg_checkpoint/checkpoint_ecg.pt
```

### 3. Download LLM Model
```bash
# Example: Download LLaMA-3.1-8B
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B

# Or use your own LLM checkpoint
```

## Configuration

Update the paths in your scripts to match your data locations:

### train.py
```python
# Line 141-150
--paraphrased_path /path/to/ecgqa/ptbxl/paraphrased/
--data_path /path/to/ecg_qa_500
--model_name /path/to/llama3.1-8b
```

### inference.py
```python
# Line 132-139
--paraphrased_path /path/to/ecgqa/ptbxl/paraphrased/
--model_name /path/to/llama3_2_1B/
```

## Run Your First Experiment

### Training
```bash
python train.py \
  --experiment_id 1001 \
  --batchsz_train 10000 \
  --batchsz_test 1000 \
  --paraphrased_path /path/to/ecgqa/ptbxl/paraphrased/ \
  --test_dataset ptb-xl \
  --model_name /path/to/llama3.1-8b \
  --question_type single-verify \
  --epoch 1 \
  --n_way 5 \
  --k_spt 5 \
  --k_qry 5 \
  --prefix_length 4 \
  --mapper_type MLP \
  --task_num 1 \
  --meta_lr 5e-4 \
  --update_lr 0.05 \
  --update_step 15 \
  --update_step_test 15 \
  --num_workers 8
```

### Inference
```bash
python inference.py \
  --experiment_id 1001 \
  --batchsz_test 10 \
  --paraphrased_path /path/to/ecgqa/ptbxl/paraphrased/ \
  --test_dataset ptb-xl \
  --model_name /path/to/llama3.1-8b \
  --question_type single-verify \
  --n_way 5 \
  --k_spt 5 \
  --k_qry 5 \
  --prefix_length 4 \
  --mapper_type MLP \
  --task_num 1 \
  --update_lr 0.05 \
  --num_workers 8 \
  --update_step 15 \
  --update_step_test 15
```

## Troubleshooting

### Common Issues

**1. "No module named 'fairseq_signals'"**
```bash
# Solution: Install fairseq-signals
git clone https://github.com/sajus2023/fairseq-signals.git
cd fairseq-signals
pip install --editable ./
```

**2. "CUDA out of memory"**
```bash
# Solution: Reduce batch size or number of workers
--batchsz_train 5000 --num_workers 4
```

**3. "File not found" errors**
```bash
# Solution: Check and update data paths
# Make sure paths in train.py and inference.py match your data location
```

**4. Import errors**
```bash
# Solution: Run verification script
python3 verify_installation.py

# Install missing packages
pip install -r requirements.txt
```

## Output Files

### Training
- **Models:** Saved to `./models/` directory
- **Logs:** Saved to `./logs/log_{experiment_id}.txt`

### Inference
- **Predictions:** Logged to `./logs/log_{experiment_id}.txt`
- **Metrics:** BERTScore, METEOR, ROUGE, BLEU scores

## Next Steps

1. **Experiment with different configurations:**
   - Try different question types: `single-verify`, `single-choose`, `single-query`, `all`
   - Adjust n-way and k-shot parameters
   - Test different LLM models

2. **Analyze results:**
   - Check log files in `./logs/`
   - Compare performance across different settings

3. **Read the documentation:**
   - [README.md](README.md) - Main documentation
   - [MIGRATION.md](MIGRATION.md) - Python 3.12 migration details
   - [CHANGES.md](CHANGES.md) - List of all changes

## Performance Tips

1. **Use GPU:** 10-100x faster than CPU
2. **Virtual Environment:** Isolate dependencies
3. **Parallel Workers:** Increase `--num_workers` for faster data loading
4. **Batch Size:** Increase for better GPU utilization (if memory allows)

## Getting Help

- **Installation issues:** Check [MIGRATION.md](MIGRATION.md)
- **Code issues:** Open an issue on GitHub
- **Original paper:** Read the CHIL 2025 paper (see [README.md](README.md))

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Monitor training in real-time
tail -f logs/log_*.txt

# Check Python version
python3 --version

# List installed packages
pip list

# Run verification
python3 verify_installation.py
```

## Example Workflow

```bash
# 1. Setup
./setup.sh

# 2. Verify
python3 verify_installation.py

# 3. Download data and models
# (Manual download from Hugging Face)

# 4. Quick test with small batch
python train.py --batchsz_train 100 --batchsz_test 10 --epoch 1

# 5. Full training
python train.py --batchsz_train 10000 --epoch 10

# 6. Inference
python inference.py --batchsz_test 100

# 7. Analyze results
cat logs/log_*.txt
```

## Resources

- **Dataset:** https://huggingface.co/datasets/jialucode/FSL_ECG_QA_Dataset/
- **Models:** https://huggingface.co/jialucode/FSL_ECG_QA/
- **Paper:** https://arxiv.org/html/2410.14464v1
- **fairseq-signals:** https://github.com/sajus2023/fairseq-signals

---

**Ready to start?** Run `./setup.sh` and you'll be up and running in minutes!
