"""Data loader for patent matching task (auto-generating candidate IDs when absent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import Sample


def _normalize_ctx_list(raw_ctx: Any) -> List[Dict[str, Any]]:
    """Ensure the context list is a list of dicts."""
    if not isinstance(raw_ctx, list):
        return []
    return [c for c in raw_ctx if isinstance(c, dict)]


def load_patent_samples(path: Path) -> List[Sample]:
    """
    Load patent matching samples from a JSON file in the 'original' format and
    auto-generate stable IDs when missing.

    ID generation (when missing):
      positive      -> p{sample_index}_{k}
      negative      -> n{sample_index}_{k}
      hard_negative -> hn{sample_index}_{k}
    """
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of samples")

    samples: List[Sample] = []

    for sample_idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {sample_idx} is not a dictionary")

        question = item.get("question", "")
        if not question:
            raise ValueError(f"Sample at index {sample_idx} missing 'question' field")

        positive_ctxs = _normalize_ctx_list(item.get("positive_ctxs", []))
        negative_ctxs = _normalize_ctx_list(item.get("negative_ctxs", []))
        hard_negative_ctxs = _normalize_ctx_list(item.get("hard_negative_ctxs", []))

        candidates: List[Dict[str, Any]] = []
        ground_truth_ids: List[str] = []

        # Process positive contexts
        for k, ctx in enumerate(positive_ctxs):
            raw_id = str(ctx.get("id", "")).strip()
            ctx_id = raw_id if raw_id else f"p{sample_idx}_{k}"
            text = str(ctx.get("text", "")).strip()
            candidates.append({
                "id": ctx_id,
                "text": text,
                "label": "positive",
                "type": "positive",
                "title": str(ctx.get("title", "")),
            })
            ground_truth_ids.append(ctx_id)

        # Process negative contexts
        for k, ctx in enumerate(negative_ctxs):
            raw_id = str(ctx.get("id", "")).strip()
            ctx_id = raw_id if raw_id else f"n{sample_idx}_{k}"
            text = str(ctx.get("text", "")).strip()
            candidates.append({
                "id": ctx_id,
                "text": text,
                "label": "negative",
                "type": "negative",
                "title": str(ctx.get("title", "")),
            })

        # Process hard negative contexts
        for k, ctx in enumerate(hard_negative_ctxs):
            raw_id = str(ctx.get("id", "")).strip()
            ctx_id = raw_id if raw_id else f"hn{sample_idx}_{k}"
            text = str(ctx.get("text", "")).strip()
            candidates.append({
                "id": ctx_id,
                "text": text,
                "label": "negative",        # classification label stays 'negative'
                "type": "hard_negative",
                "title": str(ctx.get("title", "")),
            })

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