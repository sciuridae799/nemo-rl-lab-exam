from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.utils.checkpoint_seed import seed_checkpoint_step
from nemo_rl_lab.config_resolve import resolve

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_seed_checkpoint_step_can_renumber_copy_without_touching_source(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_step = _make_checkpoint(source)

    target_step = seed_checkpoint_step(source, target, 30, target_step=0)

    assert target_step == target / "step_0"
    assert (target_step / "policy" / "weights").is_dir()
    assert source_step == source / "step_30"


def test_qa_grpo_coldstart_probe_is_weights_only_validation():
    config = resolve(
        REPO_ROOT
        / "experiments"
        / "grpo_qwen3.5-9b_qa-rl-agent_v1"
        / "config.yaml"
    )

    assert config["data"]["resume_weights_only"] is True
    assert config["data"]["weights_only_validation_only"] is True
    assert config["data"]["retrieval_diagnostic"] is False
    assert config["data"]["resume_checkpoint_step"] == 4
    assert config["grpo"]["max_num_steps"] == 3
    assert config["grpo"]["val_period"] == 3
    assert config["policy"]["megatron_cfg"]["scheduler"]["lr_decay_iters"] == 3
    assert config["env"]["qa_search"]["cfg"]["evidence_reward_scale"] == 0.0
