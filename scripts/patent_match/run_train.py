#python
# scripts/patent_match/run_train.py
#!/usr/bin/env python3
"""Train ACE on patent matching classification task with DeepSeek API (clean output)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import List
import logging

# 确保项目根目录在 sys.path 里，便于导入 scripts.* 与顶层模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 安装统一输出补丁（需在导入 tqdm 之前）
from scripts.patent_match.console_output import install_output_patch
install_output_patch(log_file_path="runs/patent_match/train.log")

import tqdm

# 兼容：某些环境下模块`tqdm`不提供`write`，注入一个回退实现
if not hasattr(tqdm, "write"):
    def _tqdm_write(msg: str = "", end: str = "\n") -> None:
        print(msg, end=end, flush=True)
    tqdm.write = _tqdm_write  # type: ignore[attr-defined]

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

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--playbook-path", default=None)
    parser.add_argument("--save-playbook", default=None)
    parser.add_argument("--max-refinement-rounds", type=int, default=2)
    parser.add_argument("--reflection-window", type=int, default=3)
    return parser.parse_args()


def _bullet_count(pb: Playbook) -> int:
    bullets = pb.bullets
    if callable(bullets):
        try:
            bullets = bullets()
        except Exception:
            return 0
    try:
        return len(bullets)
    except Exception:
        return 0


def print_step_details(step: int, total: int, result: AdapterStepResult) -> None:
    metrics = result.environment_result.metrics
    tqdm.write(f"\n{'='*80}")
    tqdm.write(f"Step {step}/{total}")
    tqdm.write(f"{'='*80}")
    tqdm.write(f"Question: {result.sample.question[:100]}...")
    try:
        context_data = json.loads(result.sample.context)
        num_candidates = len(context_data.get("candidates", []))
        tqdm.write(f"Candidates: {num_candidates}")
    except Exception:
        tqdm.write("Candidates: unknown")
    tqdm.write("\nMetrics:")
    tqdm.write(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    tqdm.write(f"  Precision: {metrics.get('precision', 0):.2%}")
    tqdm.write(f"  Recall:    {metrics.get('recall', 0):.2%}")
    tqdm.write(f"  F1 Score:  {metrics.get('f1', 0):.2%}")

    feedback = result.environment_result.feedback
    if "JSON:" in feedback:
        try:
            json_part = feedback.split("JSON:")[1].strip()
            error_data = json.loads(json_part)
            fp_neg = error_data.get("fp_from_negative_ids", [])
            fp_hard_neg = error_data.get("fp_from_hard_negative_ids", [])
            fn_pos = error_data.get("fn_positive_ids", [])
            per_id_reason = error_data.get("per_id_reason", {})
            if fp_neg or fp_hard_neg or fn_pos:
                tqdm.write("\nErrors:")
                if fp_neg:
                    tqdm.write(f"  FP (from negative): {len(fp_neg)}")
                    for fp_id in fp_neg[:2]:
                        tqdm.write(f"    - {fp_id}: {per_id_reason.get(fp_id, 'N/A')[:80]}...")
                if fp_hard_neg:
                    tqdm.write(f"  FP (from hard_negative): {len(fp_hard_neg)}")
                    for fp_id in fp_hard_neg[:2]:
                        tqdm.write(f"    - {fp_id}: {per_id_reason.get(fp_id, 'N/A')[:80]}...")
                if fn_pos:
                    tqdm.write(f"  FN (missed positive): {len(fn_pos)}")
                    for fn_id in fn_pos[:2]:
                        tqdm.write(f"    - {fn_id}: {per_id_reason.get(fn_id, 'N/A')[:80]}...")
        except Exception as e:
            tqdm.write(f"  (Could not parse error details: {e})")

    delta_ops = result.curator_output.delta.operations
    new_bullets = [op for op in delta_ops if op.type == "ADD"]
    if new_bullets:
        tqdm.write(f"\nNew bullets added: {len(new_bullets)}")
        for op in new_bullets[:2]:
            content = getattr(op, "content", "") or ""
            tqdm.write(f"  - {content[:100]}...")


def print_epoch_summary(epoch: int, results: List[AdapterStepResult]) -> None:
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

    total_adds = total_updates = total_removes = 0
    for r in results:
        for op in r.curator_output.delta.operations:
            if op.type == "ADD":
                total_adds += 1
            elif op.type == "UPDATE":
                total_updates += 1
            elif op.type == "REMOVE":
                total_removes += 1
    tqdm.write("\nPlaybook Updates:")
    tqdm.write(f"  Added:   {total_adds}")
    tqdm.write(f"  Updated: {total_updates}")
    tqdm.write(f"  Removed: {total_removes}")
    tqdm.write(f"{'#'*80}\n")


def main() -> None:
    args = parse_args()

    logger.info(f"Loading training data from {args.train_json}...")
    train_samples = load_patent_samples(Path(args.train_json))
    logger.info(f"Loaded {len(train_samples)} training samples")

    logger.info(f"\nInitializing DeepSeek client (model: {args.model})...")
    client = DeepSeekClient(
        model=args.model,
        base_url=args.deepseek_base_url,
        # 关键：传入默认采样参数
        default_temperature=args.temperature,
        default_max_new_tokens=args.max_new_tokens,
    )

    generator = Generator(llm=client, prompt_template=GENERATOR_PROMPT_PATENT_CLS)
    reflector = Reflector(llm=client, prompt_template=REFLECTOR_PROMPT_PATENT_CLS)
    curator = Curator(llm=client, prompt_template=CURATOR_PROMPT_PATENT_CLS)

    playbook = None
    if args.playbook_path:
        path = Path(args.playbook_path)
        logger.info(f"Loading playbook from {path}...")
        text = path.read_text(encoding="utf-8")
        playbook = Playbook.loads(text)
        logger.info(f"Loaded playbook with {_bullet_count(playbook)} bullets")

    adapter = OfflineAdapter(
        playbook=playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=args.max_refinement_rounds,
        reflection_window=args.reflection_window,
    )
    environment = PatentMatchEnvironment()

    logger.info(f"\nStarting training for {args.epochs} epoch(s)...")
    logger.info("Configuration:")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_new_tokens}")
    logger.info(f"  Max refinement rounds: {args.max_refinement_rounds}")
    logger.info(f"  Reflection window: {args.reflection_window}")

    total_steps = len(train_samples)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # python
    # 放在 runs_dir.mkdir(...) 之后，替换掉原来的双重 epoch 循环

    for epoch in range(1, args.epochs + 1):
        tqdm.write(f"\n{'*' * 80}")
        tqdm.write(f"EPOCH {epoch}/{args.epochs}")
        tqdm.write(f"{'*' * 80}")

        epoch_results: List[AdapterStepResult] = []
        bullet_ids_this_epoch: List[str] = []

        for step_index, sample in enumerate(
                tqdm.tqdm(train_samples, desc=f"Epoch {epoch} progress", unit="sample"),
                start=1,
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

        if getattr(adapter, "deduplicator", None):
            adapter.playbook.deduplicate(adapter.deduplicator, bullet_ids_this_epoch)

        print_epoch_summary(epoch, epoch_results)

        ckpt_path = runs_dir / f"playbook_epoch{epoch}.json"
        ckpt_path.write_text(adapter.playbook.dumps(), encoding="utf-8")
        tqdm.write(f"Saved checkpoint: {ckpt_path}")

    if args.save_playbook:
        final_path = Path(args.save_playbook)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"\nSaving final playbook to {final_path}...")
        final_path.write_text(adapter.playbook.dumps(), encoding="utf-8")
        logger.info(f"Playbook saved with {_bullet_count(adapter.playbook)} bullets")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
