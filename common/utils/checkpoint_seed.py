"""安全地把历史 checkpoint 的单个 step 播种到新 run 目录。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _link_or_copy(source: str, target: str) -> str:
    """同一文件系统优先硬链接，跨文件系统时退回复制。"""
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def seed_checkpoint_step(
    source_root: str | Path,
    target_root: str | Path,
    step: int,
) -> Path:
    """把 ``source_root/step_N`` 放入空的新 run，绝不覆盖已有 step。"""
    step = int(step)
    if step < 0:
        raise ValueError("checkpoint step 必须非负")

    source_root = Path(source_root).expanduser()
    target_root = Path(target_root).expanduser()
    if source_root.resolve() == target_root.resolve():
        raise ValueError("resume 源目录与目标目录不能相同")

    source_step = source_root / f"step_{step}"
    required = (
        source_step / "training_info.json",
        source_step / "train_dataloader.pt",
        source_step / "policy" / "weights",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("resume checkpoint 不完整：" + ", ".join(missing))

    target_root.mkdir(parents=True, exist_ok=True)
    existing_steps = sorted(target_root.glob("step_*"))
    if existing_steps:
        raise FileExistsError(
            "目标 run 已有 checkpoint，拒绝覆盖："
            + ", ".join(str(path) for path in existing_steps)
        )

    target_step = target_root / source_step.name
    shutil.copytree(source_step, target_step, copy_function=_link_or_copy)
    return target_step
