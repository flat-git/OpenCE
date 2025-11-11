# -*- coding: utf-8 -*-
"""
仅替换输出方式的补丁模块：
- 在非 TTY / PyCharm / CI 环境中禁用 tqdm 动态进度条，防止控制台刷屏。
- 统一设置控制台与可选文件日志，不改训练/推理逻辑。
- 需在入口脚本的最顶部调用 install_output_patch(...)。
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional


def _is_quiet_env() -> bool:
    """检测是否应关闭动态进度条输出。"""
    if os.environ.get("FORCE_TQDM", "") == "1":
        return False
    if os.environ.get("FORCE_QUIET", "") == "1":
        return True
    # 无交互终端/被重定向/IDE 控制台/CI 环境 => 安静模式
    return (
        not sys.stderr.isatty()
        or os.environ.get("PYCHARM_HOSTED", "") == "1"
        or os.environ.get("CI", "") == "1"
        or os.environ.get("GITHUB_ACTIONS", "") == "1"
    )


def _patch_tqdm(quiet: bool, mininterval: float = 1.5) -> None:
    """
    覆盖 tqdm 的默认行为：在安静环境禁用；在交互环境降低刷新频率，关闭动态宽度与残留。
    不改变任何训练/推理逻辑，仅影响进度显示。
    """
    try:
        import tqdm as _tqdm_mod  # type: ignore
        from tqdm import tqdm as _base_tqdm  # type: ignore
    except Exception:
        # tqdm 不存在则忽略
        return

    # 环境变量级别禁用，兜底
    if quiet:
        os.environ.setdefault("TQDM_DISABLE", "1")

    def _patched_tqdm(*args, **kwargs):
        # 仅在未显式传入 disable 时按环境决定
        kwargs.setdefault("disable", quiet)
        # 稳定输出，避免频繁重绘与残留
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("leave", False)
        kwargs.setdefault("mininterval", float(os.environ.get("TQDM_MININTERVAL", str(mininterval))))
        kwargs.setdefault("smoothing", 0)
        # 屏幕友好的简洁格式
        kwargs.setdefault(
            "bar_format",
            "{l_bar}{bar}| {n_fmt}/{total_fmt} {percentage:3.0f}% | {elapsed}<{remaining} {postfix}",
        )
        return _base_tqdm(*args, **kwargs)

    # 模块级替换（要求在其他模块 import tqdm 之前执行）
    _tqdm_mod.tqdm = _patched_tqdm  # type: ignore[attr-defined]


def _setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    统一设置日志，仅影响输出，不改业务逻辑。
    - 控制台：简洁单行格式
    - 文件（可选）：包含时间戳的详细日志
    """
    root = logging.getLogger()
    root.setLevel(level)

    # 清理现有 handler，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(fmt="%(message)s"))
    root.addHandler(console)

    if log_file:
        # 确保目录存在
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except Exception:
            pass
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setLevel(level)
        file_h.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root.addHandler(file_h)


def install_output_patch(log_file_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    入口函数：在主脚本最顶部调用。
    仅替换输出方式（日志与进度条），不改训练/推理逻辑。
    """
    quiet = _is_quiet_env()
    _setup_logging(log_file_path, level=level)
    _patch_tqdm(quiet=quiet)
    # 明确提示当前输出模式（仅一行，不会重复刷屏）
    logging.getLogger(__name__).info(
        "[output] mode=%s, log_file=%s, tty=%s, pycharm=%s, ci=%s",
        "quiet" if quiet else "interactive",
        log_file_path or "-",
        str(sys.stderr.isatty()),
        os.environ.get("PYCHARM_HOSTED", "0"),
        os.environ.get("CI", "0"),
    )