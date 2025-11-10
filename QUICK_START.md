# Quick Start Guide - Patent Matching with DeepSeek

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Setup DeepSeek API

```bash
# Set your API key
export DEEPSEEK_API_KEY="your-api-key-here"

# Optional: custom base URL
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

## Quick Test with Sample Data

### 1. Train on Sample Data (1 epoch)

```bash
cd scripts/patent_match

python run_train.py \
  --train-json sample_data.json \
  --epochs 1 \
  --save-playbook trained_playbook.json
```

Expected output:
- Per-step metrics (Accuracy, Precision, Recall, F1)
- Error examples with reasons
- Playbook updates
- Epoch summary

### 2. Evaluate on Test Data

```bash
python run_eval.py \
  --test-json sample_data.json \
  --playbook-path trained_playbook.json \
  --verbose
```

Expected output:
- Per-sample detailed results
- Error analysis
- Final aggregate metrics

## Python API Usage

### Using DeepSeekClient directly

```python
from ace import DeepSeekClient

# Initialize client
client = DeepSeekClient(
    model="deepseek-chat",
    api_key="your-key",
    timeout=30.0
)

# Generate completion
response = client.complete(
    prompt="Your prompt here",
    temperature=0.0,
    max_new_tokens=512
)

print(response.text)
```

### Loading Patent Matching Data

```python
from pathlib import Path
import sys
sys.path.insert(0, "scripts/patent_match")

from loader import load_patent_samples

# Load samples
samples = load_patent_samples(Path("data/train.json"))

print(f"Loaded {len(samples)} samples")
for sample in samples[:3]:
    print(f"Question: {sample.question[:50]}...")
```

### Running Evaluation

```python
from ace import Generator, Playbook, DeepSeekClient
from environment import PatentMatchEnvironment
from prompts import GENERATOR_PROMPT_PATENT_CLS

# Setup
client = DeepSeekClient(api_key="your-key")
generator = Generator(llm=client, prompt_template=GENERATOR_PROMPT_PATENT_CLS)
environment = PatentMatchEnvironment()
playbook = Playbook()  # or load from file

# Evaluate single sample
output = generator.generate(
    question=sample.question,
    context=sample.context,
    playbook=playbook
)

result = environment.evaluate(sample, output)
print(f"Accuracy: {result.metrics['accuracy']:.2%}")
print(f"F1: {result.metrics['f1']:.2%}")
```

## Data Format

Your JSON file should follow this structure:

```json
[
  {
    "question": "Patent question describing the invention",
    "positive_ctxs": [
      {
        "id": "pos-1",
        "text": "Relevant patent context that matches"
      }
    ],
    "negative_ctxs": [
      {
        "id": "neg-1",
        "text": "Irrelevant patent context"
      }
    ],
    "hard_negative_ctxs": [
      {
        "id": "hard-1",
        "text": "Deceptively similar but actually irrelevant"
      }
    ]
  }
]
```

## Common Options

### Training Options

```bash
python run_train.py \
  --train-json train.json \
  --epochs 3 \
  --model deepseek-chat \
  --temperature 0.0 \
  --max-new-tokens 2048 \
  --max-refinement-rounds 2 \
  --reflection-window 3 \
  --playbook-path initial_playbook.json \
  --save-playbook final_playbook.json
```

### Evaluation Options

```bash
python run_eval.py \
  --test-json test.json \
  --model deepseek-chat \
  --temperature 0.0 \
  --max-new-tokens 2048 \
  --playbook-path trained_playbook.json \
  --verbose  # for detailed per-sample output
```

## Output Interpretation

### Training Output

Each step shows:
- **Sample info**: Question snippet, candidate count
- **Metrics**: Accuracy, Precision, Recall, F1
- **Errors**: FP from negatives, FP from hard negatives, FN
- **Reasons**: One-line explanation per error
- **Playbook updates**: New bullets added

Each epoch shows:
- **Average metrics**: Across all samples
- **Operation counts**: ADD, UPDATE, REMOVE

### Evaluation Output

Final summary includes:
- **Average metrics**: Mean across samples
- **Micro-averaged metrics**: Across all candidates
- **Confusion matrix**: TP, FP, FN, TN

## Troubleshooting

### API Key Issues

```bash
# Check if key is set
echo $DEEPSEEK_API_KEY

# Set key in current session
export DEEPSEEK_API_KEY="sk-..."
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check installation
python -c "from ace import DeepSeekClient; print('OK')"
```

### Timeout Errors

Increase timeout in client initialization:
```python
client = DeepSeekClient(api_key="...", timeout=60.0)
```

## Next Steps

1. Prepare your patent matching dataset in the required format
2. Run training with multiple epochs
3. Evaluate on held-out test set
4. Analyze error patterns from output
5. Iterate on prompt templates if needed

## Documentation

- Full implementation details: `IMPLEMENTATION_SUMMARY.md`
- Architecture diagrams: `docs/architecture_patent_match.md`
- Task-specific README: `scripts/patent_match/README.md`

## Support

For issues or questions:
1. Check existing tests: `tests/test_deepseek_client.py`, `tests/test_patent_match.py`
2. Review sample data: `scripts/patent_match/sample_data.json`
3. Check prompt templates: `scripts/patent_match/prompts.py`
