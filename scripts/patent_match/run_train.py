#!/usr/bin/env python3
"""Train ACE on patent matching classification task with DeepSeek API (with tqdm progress bar)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import List

from tqdm import tqdm  # progress bar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
    DeepSeekClient,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Playbook,
    AdapterStepResult,
)

from loader import load_patent_samples
from environment import PatentMatchEnvironment
from prompts import (
    GENERATOR_PROMPT_PATENT_CLS,
    REFLECTOR_PROMPT_PATENT_CLS,
    CURATOR_PROMPT_PATENT_CLS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, help="Path to training data JSON file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument(
        "--deepseek-base-url",
        default=None,
        help="Optional DeepSeek API base URL (defaults to env DEEPSEEK_BASE_URL or https://api.deepseek.com)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--playbook-path", default=None, help="Optional path to load initial playbook")
    parser.add_argument("--save-playbook", default=None, help="Optional path to save final playbook")
    parser.add_argument(
        "--max-refinement-rounds", type=int, default=2, help="Maximum refinement rounds for reflection"
    )
    parser.add_argument(
        "--reflection-window", type=int, default=3, help="Number of recent reflections to keep in context"
    )
    return parser.parse_args()


def print_step_details(step: int, total: int, result: AdapterStepResult) -> None:
    """Print detailed training step information (uses tqdm.write to preserve progress bar)."""
    metrics = result.environment_result.metrics

    tqdm.write(f"\n{'='*80}")
    tqdm.write(f"Step {step}/{total}")
    tqdm.write(f"{'='*80}")
    tqdm.write(f"Question: {result.sample.question[:100]}...")

    # Candidate count
    try:
        context_data = json.loads(result.sample.context)
        num_candidates = len(context_data.get("candidates", []))
        tqdm.write(f"Candidates: {num_candidates}")
    except Exception:
        tqdm.write("Candidates: unknown")

    # Metrics
    tqdm.write("\nMetrics:")
    tqdm.write(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    tqdm.write(f"  Precision: {metrics.get('precision', 0):.2%}")
    tqdm.write(f"  Recall:    {metrics.get('recall', 0):.2%}")
    tqdm.write(f"  F1 Score:  {metrics.get('f1', 0):.2%}")

    # Errors
    feedback = result.environment_result.feedback
    if "JSON:" in feedback:
        try:
            json_part = feedback.split("JSON:")[1].strip()
            error_data = json.loads(json_part)

            fp_neg = error_data.get("fp_from_negative_ids", [])
            fp_hard_neg = error_data.get("fp_from_hard_negative_ids", [])
            fn_pos = error_data.get("fn_positive_ids", [])
            per_id_reason = error_data.get("per_id_reason", {})

            tqdm.write("\nErrors:")
            if fp_neg:
                tqdm.write(f"  FP (from negative): {len(fp_neg)} errors")
                for fp_id in fp_neg[:2]:
                    reason = per_id_reason.get(fp_id, "N/A")
                    tqdm.write(f"    - {fp_id}: {reason[:80]}...")

            if fp_hard_neg:
                tqdm.write(f"  FP (from hard_negative): {len(fp_hard_neg)} errors")
                for fp_id in fp_hard_neg[:2]:
                    reason = per_id_reason.get(fp_id, "N/A")
                    tqdm.write(f"    - {fp_id}: {reason[:80]}...")

            if fn_pos:
                tqdm.write(f"  FN (missed positive): {len(fn_pos)} errors")
                for fn_id in fn_pos[:2]:
                    reason = per_id_reason.get(fn_id, "N/A")
                    tqdm.write(f"    - {fn_id}: {reason[:80]}...")
        except Exception as e:
            tqdm.write(f"  (Could not parse error details: {e})")

    # New bullets (ADD operations)
    delta_ops = result.curator_output.delta.operations
    new_bullets = [op for op in delta_ops if op.type == "ADD"]
    if new_bullets:
        tqdm.write(f"\nNew bullets added: {len(new_bullets)}")
        for op in new_bullets[:2]:
            content = getattr(op, "content", None) or ""
            tqdm.write(f"  - {content[:100]}...")


def print_epoch_summary(epoch: int, results: List[AdapterStepResult]) -> None:
    """Print epoch-level summary statistics."""
    all_accuracies = [r.environment_result.metrics.get("accuracy", 0) for r in results]
    all_precisions = [r.environment_result.metrics.get("precision", 0) for r in results]
    all_recalls = [r.environment_result.metrics.get("recall", 0) for r in results]
    all_f1s = [r.environment_result.metrics.get("f1", 0) for r in results]

    tqdm.write(f"\n{'#'*80}")
    tqdm.write(f"EPOCH {epoch} SUMMARY")
    tqdm.write(f"{'#'*80}")
    tqdm.write("Average Metrics:")
    tqdm.write(f"  Accuracy:  {mean(all_accuracies):.2%}")
    tqdm.write(f"  Precision: {mean(all_precisions):.2%}")
    tqdm.write(f"  Recall:    {mean(all_recalls):.2%}")
    tqdm.write(f"  F1 Score:  {mean(all_f1s):.2%}")

    total_adds = 0
    total_updates = 0
    total_removes = 0
    for r in results:
        for op in r.curator_output.delta.operations:
            if op.type == "ADD":
                total_adds += 1
            elif op.type == "UPDATE":
                total_updates += 1
            elif op.type == "REMOVE":
                total_removes += 1

    tqdm.write("\nPlaybook Updates:")
    tqdm.write(f"  Added:   {total_adds} bullets")
    tqdm.write(f"  Updated: {total_updates} bullets")
    tqdm.write(f"  Removed: {total_removes} bullets")
    tqdm.write(f"{'#'*80}\n")


def main() -> None:
    args = parse_args()

    # Load data
    print(f"Loading training data from {args.train_json}...")
    train_samples = load_patent_samples(Path(args.train_json))
    print(f"Loaded {len(train_samples)} training samples")

    # DeepSeek client
    print(f"\nInitializing DeepSeek client (model: {args.model})...")
    client = DeepSeekClient(
        model=args.model,
        base_url=args.deepseek_base_url,
    )

    # Roles
    generator = Generator(llm=client, prompt_template=GENERATOR_PROMPT_PATENT_CLS)
    reflector = Reflector(llm=client, prompt_template=REFLECTOR_PROMPT_PATENT_CLS)
    curator = Curator(llm=client, prompt_template=CURATOR_PROMPT_PATENT_CLS)

    # Playbook (optional load)
    playbook = None
    if args.playbook_path:
        print(f"Loading playbook from {args.playbook_path}...")
        playbook = Playbook.load(Path(args.playbook_path))
        print(f"Loaded playbook with {len(playbook.bullets)} bullets")

    adapter = OfflineAdapter(
        playbook=playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=args.max_refinement_rounds,
        reflection_window=args.reflection_window,
    )

    environment = PatentMatchEnvironment()

    # Config display
    print(f"\nStarting training for {args.epochs} epoch(s)...")
    print("Configuration:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print(f"  Max refinement rounds: {args.max_refinement_rounds}")
    print(f"  Reflection window: {args.reflection_window}")

    total_steps = len(train_samples)

    for epoch in range(1, args.epochs + 1):
        tqdm.write(f"\n{'*'*80}")
        tqdm.write(f"EPOCH {epoch}/{args.epochs}")
        tqdm.write(f"{'*'*80}")

        epoch_results: List[AdapterStepResult] = []
        bullet_ids_this_epoch: List[str] = []

        # Progress bar
        for step_index, sample in enumerate(
            tqdm(train_samples, desc=f"Epoch {epoch} progress", unit="sample"), start=1
        ):
            result = adapter._process_sample(
                sample,
                environment,
                epoch=epoch,
                total_epochs=args.epochs,
                step_index=step_index,
                total_steps=total_steps,
            )
            epoch_results.append(result)
            bullet_ids_this_epoch.extend(
                op.bullet_id for op in result.curator_output.delta.operations if op.bullet_id
            )
            print_step_details(step_index, total_steps, result)

        # Dedup if available
        if getattr(adapter, "deduplicator", None):
            adapter.playbook.deduplicate(adapter.deduplicator, bullet_ids_this_epoch)

        print_epoch_summary(epoch, epoch_results)

    if args.save_playbook:
        print(f"\nSaving final playbook to {args.save_playbook}...")
        adapter.playbook.save(Path(args.save_playbook))
        print(f"Playbook saved with {len(adapter.playbook.bullets)} bullets")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()