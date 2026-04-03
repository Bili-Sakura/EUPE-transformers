#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision.transforms import v2
from transformers import AutoModel

from eupe.transformers_eupe import register_eupe_transformers


def make_transform(resize_size: int = 256):
    """Create EUPE inference preprocessing with ImageNet normalization."""

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with EUPE HF-style checkpoint")
    parser.add_argument("--model-dir", required=True, help="Directory containing config.json and model.safetensors")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    register_eupe_transformers()
    model = AutoModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values = make_transform()(image).unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    print("last_hidden_state shape:", tuple(outputs.last_hidden_state.shape))
    print("pooler_output shape:", tuple(outputs.pooler_output.shape))


if __name__ == "__main__":
    main()
