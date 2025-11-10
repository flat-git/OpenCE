"""Patent matching environment for classification evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import Sample, TaskEnvironment, EnvironmentResult, GeneratorOutput


class PatentMatchEnvironment(TaskEnvironment):
    """Environment for evaluating patent candidate classification.
    
    Computes binary classification metrics (accuracy, precision, recall, f1)
    and generates structured error examples for feedback.
    """
    
    def evaluate(
        self, sample: Sample, generator_output: GeneratorOutput
    ) -> EnvironmentResult:
        """Evaluate generator output against ground truth.
        
        Args:
            sample: Sample with context containing candidates and ground_truth_ids
            generator_output: Generated predictions with classifications
            
        Returns:
            EnvironmentResult with metrics, feedback including error examples
        """
        # Parse context to get candidates and ground truth
        try:
            context_data = json.loads(sample.context)
            candidates = context_data.get("candidates", [])
            ground_truth_ids = set(context_data.get("ground_truth_ids", []))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return EnvironmentResult(
                feedback=f"Error parsing context: {e}",
                ground_truth=sample.ground_truth,
                metrics={"error": 1.0}
            )
        
        # Parse predictions from generator output
        predictions = generator_output.raw.get("predictions", [])
        if not isinstance(predictions, list):
            return EnvironmentResult(
                feedback="Error: predictions field missing or invalid in generator output",
                ground_truth=sample.ground_truth,
                metrics={"error": 1.0}
            )
        
        # Build prediction map: id -> (label, reason)
        pred_map: Dict[str, tuple[str, str]] = {}
        for pred in predictions:
            if isinstance(pred, dict) and "id" in pred and "label" in pred:
                pred_id = str(pred["id"])
                pred_label = str(pred.get("label", "")).lower()
                pred_reason = str(pred.get("reason", "No reason provided"))
                pred_map[pred_id] = (pred_label, pred_reason)
        
        # Build candidate type map for error analysis
        candidate_type_map: Dict[str, str] = {}
        for cand in candidates:
            if isinstance(cand, dict) and "id" in cand and "type" in cand:
                candidate_type_map[str(cand["id"])] = str(cand["type"])
        
        # Calculate metrics
        predicted_positive_ids = {
            cid for cid, (label, _) in pred_map.items() if label == "positive"
        }
        
        # True positives, false positives, false negatives
        tp_ids = predicted_positive_ids & ground_truth_ids
        fp_ids = predicted_positive_ids - ground_truth_ids
        fn_ids = ground_truth_ids - predicted_positive_ids
        
        # Split FP by type (negative vs hard_negative)
        fp_from_negative_ids: List[str] = []
        fp_from_hard_negative_ids: List[str] = []
        for fp_id in fp_ids:
            cand_type = candidate_type_map.get(fp_id, "unknown")
            if cand_type == "hard_negative":
                fp_from_hard_negative_ids.append(fp_id)
            else:
                fp_from_negative_ids.append(fp_id)
        
        fn_positive_ids = list(fn_ids)
        
        # Calculate binary classification metrics
        tp = len(tp_ids)
        fp = len(fp_ids)
        fn = len(fn_ids)
        tn = len(candidates) - tp - fp - fn
        
        total = len(candidates)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
        }
        
        # Build per-id reason map for error feedback
        per_id_reason: Dict[str, str] = {}
        for pred_id, (_, reason) in pred_map.items():
            per_id_reason[pred_id] = reason
        
        # Construct feedback with structured error information
        feedback_parts = [
            f"Classification Results:",
            f"  Accuracy: {accuracy:.2%}",
            f"  Precision: {precision:.2%}",
            f"  Recall: {recall:.2%}",
            f"  F1: {f1:.2%}",
            f"",
            f"Errors:",
            f"  False Positives (from negative): {len(fp_from_negative_ids)}",
            f"  False Positives (from hard_negative): {len(fp_from_hard_negative_ids)}",
            f"  False Negatives (missed positive): {len(fn_positive_ids)}",
        ]
        
        # Add error examples
        if fp_from_negative_ids:
            feedback_parts.append(f"\nFP (negative) examples: {fp_from_negative_ids[:3]}")
            for fp_id in fp_from_negative_ids[:3]:
                feedback_parts.append(f"  - {fp_id}: {per_id_reason.get(fp_id, 'N/A')}")
        
        if fp_from_hard_negative_ids:
            feedback_parts.append(f"\nFP (hard_negative) examples: {fp_from_hard_negative_ids[:3]}")
            for fp_id in fp_from_hard_negative_ids[:3]:
                feedback_parts.append(f"  - {fp_id}: {per_id_reason.get(fp_id, 'N/A')}")
        
        if fn_positive_ids:
            feedback_parts.append(f"\nFN (positive) examples: {fn_positive_ids[:3]}")
            for fn_id in fn_positive_ids[:3]:
                feedback_parts.append(f"  - {fn_id}: {per_id_reason.get(fn_id, 'N/A')}")
        
        feedback_text = "\n".join(feedback_parts)
        
        # Append structured JSON for reflector parsing
        error_json = {
            "metrics": metrics,
            "fp_from_negative_ids": fp_from_negative_ids,
            "fp_from_hard_negative_ids": fp_from_hard_negative_ids,
            "fn_positive_ids": fn_positive_ids,
            "per_id_reason": per_id_reason,
        }
        feedback_with_json = feedback_text + "\n\nJSON:\n" + json.dumps(error_json, ensure_ascii=False, indent=2)
        
        return EnvironmentResult(
            feedback=feedback_with_json,
            ground_truth=sample.ground_truth,
            metrics=metrics,
        )
