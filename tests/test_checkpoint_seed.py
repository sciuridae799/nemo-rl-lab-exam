from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.utils.checkpoint_seed import seed_checkpoint_step


def _make_checkpoint(root: Path, step: int = 30) -> Path:
    step_dir = root / f"step_{step}"
    (step_dir / "policy" / "weights" / "iter_0000000").mkdir(parents=True)
    (step_dir / "training_info.json").write_text(
        json.dumps({"current_step": step}), encoding="utf-8"
    )
    (step_dir / "train_dataloader.pt").write_bytes(b"dataloader")
    (step_dir / "policy" / "weights" / "iter_0000000" / "common.pt").write_bytes(
        b"weights"
    )
    return step_dir


def test_seed_checkpoint_step_copies_complete_step_without_touching_source(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_step = _make_checkpoint(source)

    target_step = seed_checkpoint_step(source, target, 30)

    assert target_step == target / "step_30"
    assert (target_step / "training_info.json").is_file()
    assert (target_step / "train_dataloader.pt").read_bytes() == b"dataloader"
    assert (source_step / "training_info.json").is_file()


def test_seed_checkpoint_step_rejects_incomplete_source(tmp_path):
    source = tmp_path / "source"
    (source / "step_30").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="checkpoint 不完整"):
        seed_checkpoint_step(source, tmp_path / "target", 30)


def test_seed_checkpoint_step_never_overwrites_existing_run(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_checkpoint(source)
    _make_checkpoint(target, step=20)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        seed_checkpoint_step(source, target, 30)
