"""Data loader for patent matching task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import Sample


def load_patent_samples(path: Path) -> List[Sample]:
    """Load patent matching samples from JSON file.
    
    Expected JSON format:
    [
        {
            "question": "...",
            "positive_ctxs": [{"id": "...", "text": "..."}],
            "negative_ctxs": [{"id": "...", "text": "..."}],
            "hard_negative_ctxs": [{"id": "...", "text": "..."}]
        },
        ...
    ]
    
    Args:
        path: Path to JSON file containing patent samples
        
    Returns:
        List of Sample objects with candidates and ground_truth_ids in context
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of samples")
    
    samples: List[Sample] = []
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {idx} is not a dictionary")
        
        question = item.get("question", "")
        if not question:
            raise ValueError(f"Sample at index {idx} missing 'question' field")
        
        # Build unified candidates list with label and type
        candidates: List[Dict[str, Any]] = []
        ground_truth_ids: List[str] = []
        
        # Process positive contexts
        positive_ctxs = item.get("positive_ctxs", [])
        if not isinstance(positive_ctxs, list):
            positive_ctxs = []
        for ctx in positive_ctxs:
            if isinstance(ctx, dict) and "id" in ctx and "text" in ctx:
                ctx_id = str(ctx["id"])
                candidates.append({
                    "id": ctx_id,
                    "text": str(ctx["text"]),
                    "label": "positive",
                    "type": "positive"
                })
                ground_truth_ids.append(ctx_id)
        
        # Process negative contexts
        negative_ctxs = item.get("negative_ctxs", [])
        if not isinstance(negative_ctxs, list):
            negative_ctxs = []
        for ctx in negative_ctxs:
            if isinstance(ctx, dict) and "id" in ctx and "text" in ctx:
                candidates.append({
                    "id": str(ctx["id"]),
                    "text": str(ctx["text"]),
                    "label": "negative",
                    "type": "negative"
                })
        
        # Process hard negative contexts
        hard_negative_ctxs = item.get("hard_negative_ctxs", [])
        if not isinstance(hard_negative_ctxs, list):
            hard_negative_ctxs = []
        for ctx in hard_negative_ctxs:
            if isinstance(ctx, dict) and "id" in ctx and "text" in ctx:
                candidates.append({
                    "id": str(ctx["id"]),
                    "text": str(ctx["text"]),
                    "label": "negative",
                    "type": "hard_negative"
                })
        
        # Create context as JSON string containing candidates and ground truth
        context_data = {
            "candidates": candidates,
            "ground_truth_ids": ground_truth_ids
        }
        
        sample = Sample(
            question=question,
            context=json.dumps(context_data, ensure_ascii=False),
            ground_truth=json.dumps(ground_truth_ids, ensure_ascii=False)
        )
        samples.append(sample)
    
    return samples
