# Patent Matching Classification Task

This directory contains scripts for running the patent matching classification task using the ACE framework with DeepSeek API.

## Overview

The patent matching task involves classifying candidate patent contexts as either "positive" (relevant) or "negative" (irrelevant) given a patent question. The task uses:

- **Binary classification**: Each candidate is labeled as positive or negative
- **Four evaluation metrics**: Accuracy, Precision, Recall, F1 Score
- **Error analysis**: Distinguishes between false positives from regular negatives vs. hard negatives
- **Training process display**: Shows per-step and per-epoch progress with detailed error examples

## Files

- `loader.py` - Data loader for patent matching JSON format
- `environment.py` - Evaluation environment computing metrics and error analysis
- `prompts.py` - Task-specific prompts for Generator, Reflector, and Curator
- `run_train.py` - Training script with progress display
- `run_eval.py` - Evaluation script for testing
- `sample_data.json` - Example dataset

## Data Format

Input JSON should be a list of samples:

```json
[
  {
    "question": "Patent question describing the invention",
    "positive_ctxs": [
      {"id": "pos-1", "text": "Relevant context matching the question"}
    ],
    "negative_ctxs": [
      {"id": "neg-1", "text": "Irrelevant context"}
    ],
    "hard_negative_ctxs": [
      {"id": "hard-1", "text": "Deceptively similar but actually irrelevant"}
    ]
  }
]
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set DeepSeek API key:
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
# Optional: custom base URL
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## Training

Train the ACE system on patent classification:

```bash
cd scripts/patent_match
python run_train.py \
  --train-json sample_data.json \
  --epochs 2 \
  --model deepseek-chat \
  --temperature 0.0 \
  --max-new-tokens 2048 \
  --save-playbook trained_playbook.json
```

### Training Output

The training script displays:
- **Per-step details**: Question, candidate count, metrics (Accuracy, Precision, Recall, F1)
- **Error examples**: FP from negatives, FP from hard negatives, FN from positives (with reasons)
- **Playbook updates**: New bullets added during each step
- **Epoch summaries**: Aggregate metrics and total playbook operations

## Evaluation

Evaluate on test data with a trained (or empty) playbook:

```bash
python run_eval.py \
  --test-json sample_data.json \
  --model deepseek-chat \
  --playbook-path trained_playbook.json \
  --verbose
```

Use `--verbose` to see detailed per-sample results, or omit for compact progress display.

### Evaluation Output

- **Per-sample metrics**: Accuracy, Precision, Recall, F1 for each sample
- **Error analysis**: Detailed breakdown of FP/FN with reasons
- **Final summary**: 
  - Average metrics across samples
  - Micro-averaged metrics across all candidates
  - Confusion matrix (TP, FP, FN, TN)

## Key Features

### Classification Strategy
The prompts guide the model to:
1. **Decompose patent claims** into key technical elements
2. **Verify element-wise evidence** (not just keyword matching)
3. **Avoid traps**: Surface similarities, keyword distractions
4. **Structured output**: JSON with per-candidate predictions and reasons

### Error Tracking
The environment distinguishes:
- **FP from negatives**: Regular negative contexts mislabeled as positive
- **FP from hard negatives**: Tricky negatives mislabeled as positive (more serious)
- **FN from positives**: Missed positive contexts

### Training Process
- Each step shows immediate feedback with error examples
- Reflector analyzes error patterns and root causes
- Curator updates playbook with actionable guidelines
- Epoch summaries track overall progress

## Example Commands

**Quick test with sample data:**
```bash
# Training for 1 epoch
python run_train.py --train-json sample_data.json --epochs 1 --save-playbook pb.json

# Evaluation
python run_eval.py --test-json sample_data.json --playbook-path pb.json --verbose
```

**Production training:**
```bash
python run_train.py \
  --train-json train.json \
  --epochs 3 \
  --max-refinement-rounds 2 \
  --reflection-window 3 \
  --save-playbook final_playbook.json
```

## Minimal Changes Design

This implementation follows the "minimal changes" principle:
- **Library layer**: Only adds `DeepSeekClient` to ace module
- **Task layer**: All patent-specific logic in scripts (no ace core modifications)
- **Compatibility**: Uses existing ACE interfaces (Sample, TaskEnvironment, roles)
- **Extensibility**: Easy to adapt for other classification tasks
