"""Command-line interface for TabLDM.

Examples
--------
Train a model and save it::

    python -m tabular_ldm fit data.csv --target label --out model/ \
        --vae-epochs 100 --diffusion-epochs 300

Generate synthetic rows from a saved model::

    python -m tabular_ldm generate model/ --n 1000 --out synthetic.csv \
        --steps 50 --guidance 2.0

Evaluate synthetic data against real data::

    python -m tabular_ldm evaluate real.csv synthetic.csv --target label
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import TabularLDM
from .metrics import quality_report


def _cmd_fit(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data)
    model = TabularLDM(
        latent_dim=args.latent_dim,
        num_timesteps=args.timesteps,
        schedule=args.schedule,
        device=args.device,
    )
    model.fit(
        df,
        target_col=args.target,
        vae_epochs=args.vae_epochs,
        diffusion_epochs=args.diffusion_epochs,
        batch_size=args.batch_size,
    )
    model.save(args.out)
    print(f"Saved model to {args.out}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    model = TabularLDM.load(args.model, device=args.device)

    labels = None
    if args.label is not None:
        if model.num_classes == 0:
            print("Model is unconditional; --label ignored.", file=sys.stderr)
        else:
            labels = np.full(args.n, args.label, dtype=int)

    synth = model.generate(
        args.n,
        labels=labels,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
    )
    if labels is not None and hasattr(model, "_label_encoder"):
        col = model._target_col or "label"
        synth[col] = model._label_encoder.inverse_transform(labels)
    synth.to_csv(args.out, index=False)
    print(f"Wrote {len(synth)} rows to {args.out}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    real = pd.read_csv(args.real)
    synth = pd.read_csv(args.synthetic)
    report = quality_report(real, synth, target_col=args.target)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nOverall score: {report['overall_score']:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabular_ldm", description="TabLDM CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="Train and save a model from a CSV")
    p_fit.add_argument("data", help="Path to training CSV")
    p_fit.add_argument("--target", default=None, help="Label column for conditional mode")
    p_fit.add_argument("--out", default="model/", help="Output model directory")
    p_fit.add_argument("--latent-dim", type=int, default=32)
    p_fit.add_argument("--timesteps", type=int, default=1000)
    p_fit.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p_fit.add_argument("--vae-epochs", type=int, default=100)
    p_fit.add_argument("--diffusion-epochs", type=int, default=300)
    p_fit.add_argument("--batch-size", type=int, default=256)
    p_fit.add_argument("--device", default="auto")
    p_fit.set_defaults(func=_cmd_fit)

    p_gen = sub.add_parser("generate", help="Generate synthetic rows from a saved model")
    p_gen.add_argument("model", help="Path to saved model directory")
    p_gen.add_argument("--n", type=int, required=True, help="Number of rows to generate")
    p_gen.add_argument("--out", default="synthetic.csv", help="Output CSV path")
    p_gen.add_argument("--label", type=int, default=None, help="Class label (conditional)")
    p_gen.add_argument("--guidance", type=float, default=1.0, help="CFG scale")
    p_gen.add_argument("--steps", type=int, default=None, help="DDIM inference steps")
    p_gen.add_argument("--device", default="auto")
    p_gen.set_defaults(func=_cmd_generate)

    p_eval = sub.add_parser("evaluate", help="Score synthetic data against real data")
    p_eval.add_argument("real", help="Path to real CSV")
    p_eval.add_argument("synthetic", help="Path to synthetic CSV")
    p_eval.add_argument("--target", default=None, help="Label column")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
