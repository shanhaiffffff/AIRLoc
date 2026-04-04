

import argparse
import os

import yaml
from attrdict import AttrDict

from evaluation.train_mlp_weight_predictor_twostream import train


def _resolve_config_path(config_path: str) -> str:
    if os.path.exists(config_path):
        return config_path

    fallback_candidates = [
        os.path.join("evaluation", "configuration", "S3D", config_path),
        os.path.join("evaluation", "configuration", config_path),
    ]
    for cand in fallback_candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Config file not found: {config_path}. Tried: {', '.join([config_path] + fallback_candidates)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Two-Stream weight predictor (3D weights) with trainable map backbone (ResNet18)."
    )
    parser.add_argument("--config_file", type=str, default="evaluation/configuration/S3D/config_eval.yaml")
    parser.add_argument(
        "--weights_file",
        type=str,
        default="Data/S3D/best_weights_labels_train.json",
        help="Per-image weight labels JSON (wc/wf are 3D).",
    )
    parser.add_argument("--output_dir", type=str, default="results/weight_mlp_iou_twostream_trainable_map")

    parser.add_argument("--img_h", type=int, default=320)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--map_h", type=int, default=256)
    parser.add_argument("--map_w", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Use int flags to avoid Python 3.8 argparse boolean optional limitations
    parser.add_argument(
        "--freeze_img_backbone",
        type=int,
        default=1,
        choices=[0, 1],
        help="Freeze image backbone (ResNet50): 1=yes, 0=no. Default: 1.",
    )
    parser.add_argument(
        "--freeze_map_backbone",
        type=int,
        default=0,
        choices=[0, 1],
        help="Freeze map backbone (ResNet18): 1=yes, 0=no. Default: 0 (trainable).",
    )

    args = parser.parse_args()

    config_path = _resolve_config_path(args.config_file)
    with open(config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    train(
        config,
        weights_file=args.weights_file,
        output_dir=args.output_dir,
        img_h=args.img_h,
        img_w=args.img_w,
        map_h=args.map_h,
        map_w=args.map_w,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        pretrained=args.pretrained,
        amp=args.amp,
        freeze_backbone=False,
        freeze_img_backbone=bool(args.freeze_img_backbone),
        freeze_map_backbone=bool(args.freeze_map_backbone),
        num_weights=3,
    )


if __name__ == "__main__":
    main()

