# Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          OpenCE Framework                                │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    ACE Core (Unchanged)                         │    │
│  │  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌────────────┐     │    │
│  │  │ Playbook │  │ Adaptation│  │ Roles  │  │ Delta/Prom │     │    │
│  │  └──────────┘  └───────────┘  └────────┘  └────────────┘     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                LLM Clients (Extended)                           │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐       │    │
│  │  │DummyLLM    │  │Transformers  │  │ DeepSeekClient   │ NEW   │    │
│  │  │Client      │  │LLMClient     │  │   (httpx)        │       │    │
│  │  └────────────┘  └──────────────┘  └──────────────────┘       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │             Task: Patent Matching (All New)                     │    │
│  │                                                                  │    │
│  │  ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌───────────┐  │    │
│  │  │ loader.py│  │environment  │  │prompts.py│  │run_train/ │  │    │
│  │  │          │  │.py          │  │          │  │run_eval.py│  │    │
│  │  │ Loads    │  │ Evaluates   │  │ Task     │  │           │  │    │
│  │  │ patent   │  │ classifi-   │  │ specific │  │ Orchestr- │  │    │
│  │  │ data     │  │ cation      │  │ prompts  │  │ ation     │  │    │
│  │  └──────────┘  └─────────────┘  └──────────┘  └───────────┘  │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow - Training

```
┌──────────────────┐
│  Training JSON   │
│  - question      │
│  - positive_ctxs │
│  - negative_ctxs │
│  - hard_neg_ctxs │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  loader.py              │
│  - Unified candidates   │
│  - ground_truth_ids     │
└────────┬────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                    OfflineAdapter                           │
│                                                              │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Generator    │→ │ Environment  │→ │  Reflector     │  │
│  │  (DeepSeek)   │  │ (Metrics)    │  │  (DeepSeek)    │  │
│  └───────────────┘  └──────────────┘  └────────┬───────┘  │
│         ↑                  │                     │          │
│         │                  │                     ▼          │
│         │            ┌─────▼──────┐      ┌──────────────┐  │
│         └────────────│  Playbook  │◁─────│  Curator     │  │
│                      │            │      │  (DeepSeek)  │  │
│                      └────────────┘      └──────────────┘  │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Per-step Output     │
│  - Metrics           │
│  - Error examples    │
│  - Playbook updates  │
└──────────────────────┘
```

## Data Flow - Evaluation

```
┌──────────────────┐
│   Test JSON      │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  loader.py              │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Frozen Playbook (No Curator Updates)   │
│                                          │
│  ┌───────────────┐  ┌──────────────┐   │
│  │  Generator    │→ │ Environment  │   │
│  │  (DeepSeek)   │  │ (Metrics)    │   │
│  └───────────────┘  └──────────────┘   │
│         ↑                                │
│         │                                │
│   ┌─────┴──────┐                        │
│   │  Playbook  │ (read-only)            │
│   └────────────┘                        │
└─────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Per-sample Output   │
│  - Metrics           │
│  - Error analysis    │
│  - Final summary     │
└──────────────────────┘
```

## Classification Output Format

### Generator Output (JSON)
```json
{
  "reasoning": "Step-by-step element-wise analysis",
  "final_answer": "Summary: Selected positive IDs: [pos-1]",
  "bullet_ids": ["bullet-1", "bullet-2"],
  "predictions": [
    {
      "id": "pos-1",
      "label": "positive",
      "reason": "All claim elements verified with evidence"
    },
    {
      "id": "neg-1",
      "label": "negative",
      "reason": "Missing key technical element X"
    },
    {
      "id": "hard-1",
      "label": "negative",
      "reason": "Similar terms but lacks coupling adjustment mechanism"
    }
  ]
}
```

### Environment Feedback (with embedded JSON)
```
Classification Results:
  Accuracy: 66.67%
  Precision: 50.00%
  Recall: 100.00%
  F1: 66.67%

Errors:
  False Positives (from negative): 0
  False Positives (from hard_negative): 1
  False Negatives (missed positive): 0

FP (hard_negative) examples: ['hard-1']
  - hard-1: Tricked by keyword similarity

JSON:
{
  "metrics": {"accuracy": 0.67, "precision": 0.5, "recall": 1.0, "f1": 0.67},
  "fp_from_negative_ids": [],
  "fp_from_hard_negative_ids": ["hard-1"],
  "fn_positive_ids": [],
  "per_id_reason": {
    "pos-1": "Correct positive",
    "hard-1": "Tricked by keyword similarity"
  }
}
```

## Minimal Changes Principle

### Modified Files (3 lines total)
```
ace/__init__.py:
  + from .llm_deepseek import DeepSeekClient
  + "DeepSeekClient",  # in __all__

requirements.txt:
  + httpx>=0.27.0
```

### New Library File (1 file)
```
ace/llm_deepseek.py (135 lines)
  - DeepSeekClient class
  - Implements LLMClient interface
  - httpx for REST API calls
  - Retry and timeout logic
```

### New Task Files (8 files, all in scripts/patent_match/)
```
loader.py (111 lines) - Data loading
environment.py (168 lines) - Evaluation
prompts.py (158 lines) - Task prompts
run_train.py (285 lines) - Training script
run_eval.py (256 lines) - Evaluation script
README.md (152 lines) - Documentation
sample_data.json (44 lines) - Example data
__init__.py (1 line) - Package marker
```

### New Test Files (2 files)
```
tests/test_deepseek_client.py (159 lines) - 9 tests
tests/test_patent_match.py (286 lines) - 7 tests
```

## Key Design Decisions

1. **Library Extension**: DeepSeekClient as new LLMClient, not replacement
2. **Task Isolation**: All patent-specific logic in scripts/, not in ace core
3. **Output Extension**: Use GeneratorOutput.raw for task-specific fields
4. **Feedback Structure**: Embed JSON in text for structured error data
5. **Testing Strategy**: Mock httpx for unit tests, no live API calls needed
6. **Configuration**: Environment variables for API credentials
7. **Error Tracking**: Distinguish regular FP from hard negative FP
8. **Progress Display**: Rich per-step and per-epoch output for training visibility
