# scripts/patent_match/run_eval.py
#!/usr/bin/env python3
"""Evaluate ACE on patent matching classification task with frozen playbook."""

from __future__ import annotations

import argparse
import json
import sys
import datetime
import logging
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

# 全局 logger（输出方式由 console_output 安装）
logger = logging.getLogger("eval")


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
    parser.add_argument(
        "--save-results",
        default="runs/eval_results.txt",
        help="Path to write evaluation summary"
    )
    parser.add_argument(
        "--log-dir",
        default="D:/Project/PyCharmProject/OpenCE/logs",
        help="Directory to auto-save timestamped log file",
    )
    return parser.parse_args()


def _init_logging(log_dir: Path) -> Path:
    """
    使用 console_output 统一输出：创建时间戳日志文件并安装 logging/tqdm 补丁。
    仅替换输出方式，不改评估逻辑。
    """
    from scripts.patent_match.console_output import install_output_patch

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eval_{ts}.log"

    install_output_patch(log_file_path=str(log_path))
    logger.info(f"[log] 日志文件: {log_path}")
    return log_path


def _bullet_count(pb: Playbook) -> int:
    """兼容 bullets 为列表或可调用方法的情况，返回要点数量。"""
    bullets = getattr(pb, "bullets", None)
    if bullets is None:
        return 0
    try:
        bullets = bullets() if callable(bullets) else bullets
    except Exception:
        return 0
    try:
        return len(bullets)
    except Exception:
        return 0


def evaluate_sample(
    sample: Sample,
    generator: Generator,
    environment: PatentMatchEnvironment,
    playbook: Playbook,
) -> Tuple[GeneratorOutput, EnvironmentResult]:
    generator_output = generator.generate(
        question=sample.question,
        context=sample.context,
        playbook=playbook,
        reflection=None,
    )
    env_result = environment.evaluate(sample, generator_output)
    return generator_output, env_result


def print_sample_details(idx: int, total: int, sample: Sample, env_result: EnvironmentResult) -> None:
    metrics = env_result.metrics

    logger.info("\n" + "=" * 80)
    logger.info(f"Sample {idx}/{total}")
    logger.info("=" * 80)
    logger.info(f"Question: {sample.question[:100]}...")

    try:
        context_data = json.loads(sample.context)
        num_candidates = len(context_data.get("candidates", []))
        logger.info(f"Candidates: {num_candidates}")
    except Exception:
        logger.info("Candidates: unknown")

    logger.info("\nMetrics:")
    logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.2%}")
    logger.info(f"  Recall:    {metrics.get('recall', 0):.2%}")
    logger.info(f"  F1 Score:  {metrics.get('f1', 0):.2%}")

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
                logger.info("\nErrors:")
                if fp_neg:
                    logger.info(f"  FP (from negative): {len(fp_neg)} errors")
                    for fp_id in fp_neg[:2]:
                        reason = per_id_reason.get(fp_id, "N/A")
                        logger.info(f"    - {fp_id}: {reason[:80]}...")
                if fp_hard_neg:
                    logger.info(f"  FP (from hard_negative): {len(fp_hard_neg)} errors")
                    for fp_id in fp_hard_neg[:2]:
                        reason = per_id_reason.get(fp_id, "N/A")
                        logger.info(f"    - {fp_id}: {reason[:80]}...")
                if fn_pos:
                    logger.info(f"  FN (missed positive): {len(fn_pos)} errors")
                    for fn_id in fn_pos[:2]:
                        reason = per_id_reason.get(fn_id, "N/A")
                        logger.info(f"    - {fn_id}: {reason[:80]}...")
        except Exception as e:
            logger.info(f"  (Could not parse error details: {e})")


def print_final_summary(results: List[EnvironmentResult]) -> None:
    all_accuracies = [r.metrics.get("accuracy", 0) for r in results]
    all_precisions = [r.metrics.get("precision", 0) for r in results]
    all_recalls = [r.metrics.get("recall", 0) for r in results]
    all_f1s = [r.metrics.get("f1", 0) for r in results]

    logger.info("\n" + "#" * 80)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("#" * 80)
    logger.info(f"Samples evaluated: {len(results)}")
    logger.info("\nAggregate Metrics:")
    logger.info(f"  Accuracy:  {mean(all_accuracies):.2%}")
    logger.info(f"  Precision: {mean(all_precisions):.2%}")
    logger.info(f"  Recall:    {mean(all_recalls):.2%}")
    logger.info(f"  F1 Score:  {mean(all_f1s):.2%}")

    total_tp = sum(r.metrics.get("tp", 0) for r in results)
    total_fp = sum(r.metrics.get("fp", 0) for r in results)
    total_fn = sum(r.metrics.get("fn", 0) for r in results)
    total_tn = sum(r.metrics.get("tn", 0) for r in results)
    total = total_tp + total_fp + total_fn + total_tn

    micro_accuracy = (total_tp + total_tn) / total if total > 0 else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    logger.info("\nMicro-Averaged Metrics (across all candidates):")
    logger.info(f"  Accuracy:  {micro_accuracy:.2%}")
    logger.info(f"  Precision: {micro_precision:.2%}")
    logger.info(f"  Recall:    {micro_recall:.2%}")
    logger.info(f"  F1 Score:  {micro_f1:.2%}")
    logger.info("\nConfusion Matrix (total):")
    logger.info(f"  TP: {int(total_tp)}, FP: {int(total_fp)}")
    logger.info(f"  FN: {int(total_fn)}, TN: {int(total_tn)}")
    logger.info("#" * 80 + "\n")


def _write_eval_results(results: List[EnvironmentResult], out_path: Path) -> None:
    all_accuracies = [r.metrics.get("accuracy", 0) for r in results]
    all_precisions = [r.metrics.get("precision", 0) for r in results]
    all_recalls = [r.metrics.get("recall", 0) for r in results]
    all_f1s = [r.metrics.get("f1", 0) for r in results]

    macro_acc = mean(all_accuracies) if all_accuracies else 0.0
    macro_prec = mean(all_precisions) if all_precisions else 0.0
    macro_rec = mean(all_recalls) if all_recalls else 0.0
    macro_f1 = mean(all_f1s) if all_f1s else 0.0

    total_tp = sum(r.metrics.get("tp", 0) for r in results)
    total_fp = sum(r.metrics.get("fp", 0) for r in results)
    total_fn = sum(r.metrics.get("fn", 0) for r in results)
    total_tn = sum(r.metrics.get("tn", 0) for r in results)
    total = total_tp + total_fp + total_fn + total_tn

    micro_accuracy = (total_tp + total_tn) / total if total > 0 else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("################################################################################\n")
        f.write("FINAL EVALUATION RESULTS\n")
        f.write("################################################################################\n")
        f.write(f"Samples evaluated: {len(results)}\n\n")

        f.write("Aggregate Metrics (macro):\n")
        f.write(f"  Accuracy:  {macro_acc:.2%}\n")
        f.write(f"  Precision: {macro_prec:.2%}\n")
        f.write(f"  Recall:    {macro_rec:.2%}\n")
        f.write(f"  F1 Score:  {macro_f1:.2%}\n\n")

        f.write("Micro-Averaged Metrics (across all candidates):\n")
        f.write(f"  Accuracy:  {micro_accuracy:.2%}\n")
        f.write(f"  Precision: {micro_precision:.2%}\n")
        f.write(f"  Recall:    {micro_recall:.2%}\n")
        f.write(f"  F1 Score:  {micro_f1:.2%}\n\n")

        f.write("Confusion Matrix (total):\n")
        f.write(f"  TP: {int(total_tp)}, FP: {int(total_fp)}\n")
        f.write(f"  FN: {int(total_fn)}, TN: {int(total_tn)}\n")
        f.write("################################################################################\n")


def main() -> None:
    args = parse_args()
    log_path = _init_logging(Path(args.log_dir))

    logger.info(f"Loading test data from {args.test_json}...")
    test_samples = load_patent_samples(Path(args.test_json))
    logger.info(f"Loaded {len(test_samples)} test samples")

    logger.info(f"\nInitializing DeepSeek client (model: {args.model})...")
    client = DeepSeekClient(
        model=args.model,
        base_url=args.deepseek_base_url,
    )

    playbook = Playbook()
    if args.playbook_path:
        p = Path(args.playbook_path)
        logger.info(f"Loading playbook from {p}...")
        text = p.read_text(encoding="utf-8")
        playbook = Playbook.loads(text)
        logger.info(f"Loaded playbook with {_bullet_count(playbook)} bullets")
    else:
        logger.info("No playbook provided, using empty playbook")

    generator = Generator(
        llm=client,
        prompt_template=GENERATOR_PROMPT_PATENT_CLS,
    )

    environment = PatentMatchEnvironment()

    logger.info(f"\nStarting evaluation on {len(test_samples)} samples...")
    logger.info("Configuration:")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max tokens: {args.max_new_tokens}")

    results: List[EnvironmentResult] = []

    for idx, sample in enumerate(test_samples, 1):
        generator_output, env_result = evaluate_sample(
            sample, generator, environment, playbook
        )
        results.append(env_result)

        if args.verbose:
            print_sample_details(idx, len(test_samples), sample, env_result)
        else:
            metrics = env_result.metrics
            logger.info(
                f"Sample {idx}/{len(test_samples)}: "
                f"Acc={metrics.get('accuracy', 0):.2%}, "
                f"P={metrics.get('precision', 0):.2%}, "
                f"R={metrics.get('recall', 0):.2%}, "
                f"F1={metrics.get('f1', 0):.2%}"
            )

    print_final_summary(results)
    _write_eval_results(results, Path(args.save_results))
    logger.info(f"Saved evaluation summary to {args.save_results}")
    logger.info("Evaluation complete!")
    logger.info(f"日志已保存到: {log_path}")


if __name__ == "__main__":
    main()
