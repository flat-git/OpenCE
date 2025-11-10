#!/usr/bin/env python3
"""Evaluate ACE on patent matching classification task with frozen playbook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
    DeepSeekClient,
    Generator,
    Playbook,
    Sample,
    GeneratorOutput,
    EnvironmentResult,
)

from loader import load_patent_samples
from environment import PatentMatchEnvironment
from prompts import GENERATOR_PROMPT_PATENT_CLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-json",
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek model name"
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=None,
        help="Optional DeepSeek API base URL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--playbook-path",
        default=None,
        help="Optional path to load trained playbook (empty playbook if not provided)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample results"
    )
    return parser.parse_args()


def evaluate_sample(
    sample: Sample,
    generator: Generator,
    environment: PatentMatchEnvironment,
    playbook: Playbook,
) -> Tuple[GeneratorOutput, EnvironmentResult]:
    """Evaluate a single sample."""
    generator_output = generator.generate(
        question=sample.question,
        context=sample.context,
        playbook=playbook,
        reflection=None,  # No reflection in eval mode
    )
    
    env_result = environment.evaluate(sample, generator_output)
    return generator_output, env_result


def print_sample_details(idx: int, total: int, sample: Sample, env_result: EnvironmentResult) -> None:
    """Print detailed per-sample evaluation results."""
    metrics = env_result.metrics
    
    print(f"\n{'='*80}")
    print(f"Sample {idx}/{total}")
    print(f"{'='*80}")
    print(f"Question: {sample.question[:100]}...")
    
    # Parse context to get candidate count
    try:
        context_data = json.loads(sample.context)
        num_candidates = len(context_data.get("candidates", []))
        print(f"Candidates: {num_candidates}")
    except:
        print(f"Candidates: unknown")
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    print(f"  Precision: {metrics.get('precision', 0):.2%}")
    print(f"  Recall:    {metrics.get('recall', 0):.2%}")
    print(f"  F1 Score:  {metrics.get('f1', 0):.2%}")
    
    # Parse and print error examples
    feedback = env_result.feedback
    if "JSON:" in feedback:
        try:
            json_part = feedback.split("JSON:")[1].strip()
            error_data = json.loads(json_part)
            
            fp_neg = error_data.get("fp_from_negative_ids", [])
            fp_hard_neg = error_data.get("fp_from_hard_negative_ids", [])
            fn_pos = error_data.get("fn_positive_ids", [])
            per_id_reason = error_data.get("per_id_reason", {})
            
            if fp_neg or fp_hard_neg or fn_pos:
                print(f"\nErrors:")
                if fp_neg:
                    print(f"  FP (from negative): {len(fp_neg)} errors")
                    for fp_id in fp_neg[:2]:
                        reason = per_id_reason.get(fp_id, "N/A")
                        print(f"    - {fp_id}: {reason[:80]}...")
                
                if fp_hard_neg:
                    print(f"  FP (from hard_negative): {len(fp_hard_neg)} errors")
                    for fp_id in fp_hard_neg[:2]:
                        reason = per_id_reason.get(fp_id, "N/A")
                        print(f"    - {fp_id}: {reason[:80]}...")
                
                if fn_pos:
                    print(f"  FN (missed positive): {len(fn_pos)} errors")
                    for fn_id in fn_pos[:2]:
                        reason = per_id_reason.get(fn_id, "N/A")
                        print(f"    - {fn_id}: {reason[:80]}...")
        except Exception as e:
            print(f"  (Could not parse error details: {e})")


def print_final_summary(results: List[EnvironmentResult]) -> None:
    """Print final aggregate evaluation results."""
    # Calculate average metrics
    all_accuracies = [r.metrics.get("accuracy", 0) for r in results]
    all_precisions = [r.metrics.get("precision", 0) for r in results]
    all_recalls = [r.metrics.get("recall", 0) for r in results]
    all_f1s = [r.metrics.get("f1", 0) for r in results]
    
    print(f"\n{'#'*80}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'#'*80}")
    print(f"Samples evaluated: {len(results)}")
    print(f"\nAggregate Metrics:")
    print(f"  Accuracy:  {mean(all_accuracies):.2%} (avg)")
    print(f"  Precision: {mean(all_precisions):.2%} (avg)")
    print(f"  Recall:    {mean(all_recalls):.2%} (avg)")
    print(f"  F1 Score:  {mean(all_f1s):.2%} (avg)")
    
    # Calculate total TP, FP, FN, TN for micro-averaged metrics
    total_tp = sum(r.metrics.get("tp", 0) for r in results)
    total_fp = sum(r.metrics.get("fp", 0) for r in results)
    total_fn = sum(r.metrics.get("fn", 0) for r in results)
    total_tn = sum(r.metrics.get("tn", 0) for r in results)
    total = total_tp + total_fp + total_fn + total_tn
    
    micro_accuracy = (total_tp + total_tn) / total if total > 0 else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    print(f"\nMicro-Averaged Metrics (across all candidates):")
    print(f"  Accuracy:  {micro_accuracy:.2%}")
    print(f"  Precision: {micro_precision:.2%}")
    print(f"  Recall:    {micro_recall:.2%}")
    print(f"  F1 Score:  {micro_f1:.2%}")
    print(f"\nConfusion Matrix (total):")
    print(f"  TP: {int(total_tp)}, FP: {int(total_fp)}")
    print(f"  FN: {int(total_fn)}, TN: {int(total_tn)}")
    print(f"{'#'*80}\n")


def main() -> None:
    args = parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_json}...")
    test_samples = load_patent_samples(Path(args.test_json))
    print(f"Loaded {len(test_samples)} test samples")
    
    # Initialize DeepSeek client
    print(f"\nInitializing DeepSeek client (model: {args.model})...")
    client = DeepSeekClient(
        model=args.model,
        base_url=args.deepseek_base_url,
    )
    
    # Load playbook if provided
    playbook = Playbook()
    if args.playbook_path:
        print(f"Loading playbook from {args.playbook_path}...")
        playbook = Playbook.load(Path(args.playbook_path))
        print(f"Loaded playbook with {len(playbook.bullets)} bullets")
    else:
        print("No playbook provided, using empty playbook")
    
    # Initialize generator (no reflector/curator in eval mode)
    generator = Generator(
        llm=client,
        prompt_template=GENERATOR_PROMPT_PATENT_CLS,
    )
    
    # Initialize environment
    environment = PatentMatchEnvironment()
    
    # Evaluation loop
    print(f"\nStarting evaluation on {len(test_samples)} samples...")
    print(f"Configuration:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_new_tokens}")
    
    results: List[EnvironmentResult] = []
    
    for idx, sample in enumerate(test_samples, 1):
        generator_output, env_result = evaluate_sample(
            sample, generator, environment, playbook
        )
        results.append(env_result)
        
        if args.verbose:
            print_sample_details(idx, len(test_samples), sample, env_result)
        else:
            # Print progress without details
            metrics = env_result.metrics
            print(f"Sample {idx}/{len(test_samples)}: "
                  f"Acc={metrics.get('accuracy', 0):.2%}, "
                  f"P={metrics.get('precision', 0):.2%}, "
                  f"R={metrics.get('recall', 0):.2%}, "
                  f"F1={metrics.get('f1', 0):.2%}")
    
    # Print final summary
    print_final_summary(results)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
