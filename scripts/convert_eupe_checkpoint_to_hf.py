#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file

from eupe.transformers_eupe import EUPE_VIT_PRESETS, EupeViTConfig


DEFAULT_REPOS = {
    "t": "facebook/EUPE-ViT-T",
    "s": "facebook/EUPE-ViT-S",
    "b": "facebook/EUPE-ViT-B",
}


def infer_size_from_repo_id(repo_id: str) -> str:
    """Infer ViT size token (t/s/b) from a Hugging Face repository id."""

    repo_id_low = repo_id.lower()
    if "vit-t" in repo_id_low:
        return "t"
    if "vit-b" in repo_id_low:
        return "b"
    return "s"


def select_checkpoint_filename(repo_id: str) -> str:
    """Select a likely checkpoint filename from a Hugging Face model repository."""

    files = list_repo_files(repo_id=repo_id)
    safe_candidates = [f for f in files if f.endswith(".safetensors")]
    if safe_candidates:
        safe_candidates.sort()
        return safe_candidates[0]

    candidates = [
        f
        for f in files
        if f.endswith((".pth", ".pt", ".bin", ".ckpt")) and "optimizer" not in f.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found in {repo_id}")
    candidates.sort()
    return candidates[0]


def extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    """Extract a plain tensor state_dict from common checkpoint container formats."""

    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break

    if not isinstance(raw, dict):
        raise TypeError("Unsupported checkpoint format")

    saw_non_teacher_key = False
    for k in raw.keys():
        if not isinstance(k, str) or not k.startswith("teacher."):
            saw_non_teacher_key = True
            break

    if not saw_non_teacher_key and raw:
        raw = {k.replace("teacher.", "", 1): v for k, v in raw.items()}
    elif "teacher" in raw and isinstance(raw["teacher"], dict):
        raw = raw["teacher"]

    out = {}
    for k, v in raw.items():
        if not isinstance(v, torch.Tensor):
            continue
        key = k[7:] if k.startswith("module.") else k
        out[key] = v.detach().cpu().contiguous()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EUPE checkpoint to Hugging Face Transformers layout")
    parser.add_argument("-S", "--model-size", choices=["t", "s", "b"], default=None)
    parser.add_argument("--repo-id", default=None, help="Hugging Face model repo id")
    parser.add_argument("--filename", default=None, help="Checkpoint filename inside repo")
    parser.add_argument(
        "--allow-unsafe-torch-load",
        action="store_true",
        help="Allow loading non-safetensors checkpoints via torch.load (unsafe for untrusted files)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for HF artifacts")
    args = parser.parse_args()

    repo_id = args.repo_id or DEFAULT_REPOS[args.model_size]
    ckpt_name = args.filename or select_checkpoint_filename(repo_id)
    size = args.model_size or infer_size_from_repo_id(repo_id)

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
    if ckpt_name.endswith(".safetensors"):
        raw = load_file(ckpt_path, device="cpu")
    else:
        if not args.allow_unsafe_torch_load:
            raise ValueError(
                "Selected checkpoint is not .safetensors. "
                "Re-run with --allow-unsafe-torch-load only if you trust the source checkpoint."
            )
        raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(raw)

    preset_key = f"vit{size}16"
    config = EupeViTConfig(**EUPE_VIT_PRESETS[preset_key].config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(out_dir / "model.safetensors"))
    config.save_pretrained(out_dir)

    print(f"Converted: {repo_id}/{ckpt_name} -> {out_dir}")
    print(f"Artifacts: {out_dir / 'model.safetensors'}, {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
