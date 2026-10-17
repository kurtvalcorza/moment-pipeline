"""Diagnose whether a MOMENT checkpoint's reconstruction head reconstructs its input.

See RFC 0001, Amendment 1. The check needs no data set: a clean, unpadded 512-step sine (period 32)
plus a small trend goes through upstream's documented call,
``model(x_enc=..., input_mask=..., mask=...).reconstruction``, twice:

* **unmasked** (nothing hidden): the Pearson correlation between input and reconstruction. A
  working head gives about 0.99; the released MOMENT-1-base head gave -0.005 in the
  tutorial-triage review of 2026-10-15.
* **one interior patch hidden** (positions 384-391): the masked-patch MAE beside linear
  interpolation and the mean predictor.

By default it loads this repository's pinned, digest-verified checkpoint through the package
(``weights/<MODEL_KEY>/``, fetched on first use). ``--hub-model`` with ``--hub-revision`` instead
fetches another checkpoint's ``config.json`` and ``model.safetensors`` (never ``.bin``) into the
Hugging Face cache and runs the same check, for comparison only: that path is outside the pinned
supply chain.

Usage (CPU is enough)::

    python scripts/diagnose_reconstruction_head.py [--out result.json]
    python scripts/diagnose_reconstruction_head.py --hub-model AutonLab/MOMENT-1-base \\
        --hub-revision 9fea447e740eb968a9e8d80c7562ae122bdb5dde
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQ = 512
HIDDEN = slice(384, 392)


def sine() -> np.ndarray:
    t = np.arange(SEQ, dtype=float)
    return np.sin(2 * np.pi * t / 32.0) + 0.002 * t


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def load(args: argparse.Namespace):
    if args.hub_model:
        from huggingface_hub import snapshot_download
        from momentfm import MOMENTPipeline

        path = snapshot_download(
            args.hub_model,
            revision=args.hub_revision,
            allow_patterns=["config.json", "model.safetensors"],
        )
        model = MOMENTPipeline.from_pretrained(path, model_kwargs={"task_name": "reconstruction"})
        model.init()
        model.eval()
        return model, {
            "model": args.hub_model,
            "revision": args.hub_revision,
            "path": "hub (comparison only)",
        }
    sys.path.insert(0, str(ROOT / "src"))
    from moment_pipeline.model import DEFAULT_WEIGHTS_DIR, load_moment, stage_missing_files

    stage_missing_files(DEFAULT_WEIGHTS_DIR, allow_download=True)
    loaded = load_moment(task="reconstruction", weights_dir=DEFAULT_WEIGHTS_DIR, device="cpu")
    return loaded.pipeline, {
        "model": loaded.identity.name,
        "revision": loaded.identity.revision,
        "path": "pinned, digest-verified snapshot",
    }


def run(model) -> dict:
    import torch

    x = sine()
    ones = np.ones(SEQ)

    def forward(mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(
                x_enc=torch.as_tensor(x[None, None, :], dtype=torch.float32),
                input_mask=torch.as_tensor(ones[None, :], dtype=torch.long),
                mask=torch.as_tensor(mask[None, :], dtype=torch.long),
            )
        return out.reconstruction.numpy()[0, 0].astype(float)

    unmasked = forward(ones)
    mask = ones.copy()
    mask[HIDDEN] = 0
    masked = forward(mask)
    idx = np.arange(SEQ)
    visible = mask == 1
    linear = np.interp(idx, idx[visible], x[visible])
    return {
        "unmasked_pearson": pearson(x, unmasked),
        "masked_patch_mae_model": float(np.abs(masked[HIDDEN] - x[HIDDEN]).mean()),
        "masked_patch_mae_linear": float(np.abs(linear[HIDDEN] - x[HIDDEN]).mean()),
        "masked_patch_mae_mean_predictor": float(np.abs(x[visible].mean() - x[HIDDEN]).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--hub-model", help="compare another Hub checkpoint (outside the pinned supply chain)"
    )
    parser.add_argument("--hub-revision", help="immutable commit of --hub-model")
    parser.add_argument("--out", type=Path, help="write the result as JSON here")
    args = parser.parse_args()
    if args.hub_model and not args.hub_revision:
        parser.error("--hub-model needs --hub-revision (an immutable commit)")
    model, identity = load(args)
    result = {**identity, **run(model)}
    result["head_reconstructs"] = bool(
        result["unmasked_pearson"] > 0.95
        and result["masked_patch_mae_model"] < result["masked_patch_mae_mean_predictor"]
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
