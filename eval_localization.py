

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from attrdict import AttrDict
from PIL import Image
from torchvision import transforms
import tqdm

from modules.depth.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from data_utils.data_utils import LocalizationDataset
from data_utils.data_utils_zind import LocalizationDataset as LocalizationDatasetZind
from utils.data_loader_helper import load_scene_data
from utils.localization_utils import (
    localize,
    finalize_localization,
    get_ray_from_depth,
    get_ray_from_semantics,
)
from evaluation.candidate_extractor import extract_top_k_locations
from evaluation.room_predictor import predict_room_and_get_polygons
from evaluation.result_handler import calculate_recalls

from evaluation.eval_localization_iou import (
    seed_everything,
    localize_iou,
    combine_prob_volumes_3,
    refine_and_select_best_candidate_iou,
    angular_difference_deg,
)

from evaluation.geom_invdepth_piecewise_planar_fusedlasso import geom_piecewise_planar_inverse_depth_smoothing

from evaluation.train_mlp_weight_predictor_twostream import WeightClassifierTwoStream


WC_FALLBACK = [0.55, 0.40, 0.05]
WF_FALLBACK = [0.20, 0.70, 0.10]


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


def _resolve_checkpoint_path(ckpt_path: str) -> str:
    if os.path.exists(ckpt_path):
        return ckpt_path
    fallback_candidates = [
        os.path.join("results", ckpt_path),
        os.path.join("results", "weight_mlp_iou_twostream_trainable_map", ckpt_path),
        os.path.join("results", "weight_mlp", ckpt_path),
    ]
    for cand in fallback_candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}. Tried: {', '.join([ckpt_path] + fallback_candidates)}"
    )


def _normalize_scene_name(scene_name: str, is_zind: bool) -> str:
    if is_zind:
        return scene_name
    if "floor" not in scene_name:
        return f"scene_{int(scene_name.split('_')[1])}"
    return scene_name


def _load_map_tensor_cached(dataset, scene_name_raw: str, map_h: int, map_w: int, cache: dict):
    """Load floorplan_semantic.png as normalized tensor (3,map_h,map_w). Cache per scene."""
    if scene_name_raw in cache:
        return cache[scene_name_raw]

    base_dir = getattr(dataset, "data_dir", None)
    if base_dir is None:
        raise AttributeError("Dataset has no attribute 'data_dir'; cannot locate floorplan_semantic.png")

    potential_path = os.path.join(base_dir, scene_name_raw, "floorplan_semantic.png")
    if not os.path.exists(potential_path):
        potential_path = os.path.join(base_dir, "..", "raw_S3D_perspective", scene_name_raw, "floorplan_semantic.png")

    if not os.path.exists(potential_path):
        raise FileNotFoundError(f"Map not found for {scene_name_raw}. Checked: {potential_path}")

    map_img = Image.open(potential_path).convert("RGB")
    map_tensor = transforms.ToTensor()(map_img)  # CHW, 0..1
    map_tensor = F.interpolate(map_tensor.unsqueeze(0), size=(map_h, map_w), mode="nearest").squeeze(0)

    map_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    map_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    map_tensor = (map_tensor - map_mean) / map_std

    cache[scene_name_raw] = map_tensor
    return map_tensor


def _preprocess_query_img(data_ref_img: np.ndarray, img_h: int, img_w: int) -> torch.Tensor:
    """data_ref_img is CHW in [0,1]. Return normalized torch tensor CHW."""
    img = torch.tensor(data_ref_img, dtype=torch.float32)
    if img.shape[1] != img_h or img.shape[2] != img_w:
        img = F.interpolate(img.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False).squeeze(0)

    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - img_mean) / img_std
    return img


def _predict_weights(model: WeightClassifierTwoStream, img: torch.Tensor, map_tensor: torch.Tensor):
    """Predict wc, wf as python lists length 3."""
    model.eval()
    with torch.no_grad():
        wc_logits, wf_logits = model(img.unsqueeze(0), map_tensor.unsqueeze(0))
        wc = torch.softmax(wc_logits, dim=1).squeeze(0)
        wf = torch.softmax(wf_logits, dim=1).squeeze(0)

    wc = wc.detach().cpu().numpy().astype(np.float32)
    wf = wf.detach().cpu().numpy().astype(np.float32)

    wc = np.clip(wc, 0.0, None)
    wf = np.clip(wf, 0.0, None)
    wc = wc / (wc.sum() + 1e-12)
    wf = wf / (wf.sum() + 1e-12)

    return wc.tolist(), wf.tolist()


def get_predicted_rays_with_smoothing(model, img_torch, mask_torch, config, use_gt, gt_data):
    """Like evaluation.eval_localization_iou.get_predicted_rays, but with depth smoothing."""
    if not use_gt:
        with torch.no_grad():
            if "depth" in gt_data:
                pred_depths, _, _ = model.encoder(img_torch, mask_torch)
                pred_depths_np = pred_depths.squeeze(0).cpu().numpy()

                lam = float(getattr(config, "smooth_lambda", 0.0) or 0.0)
                if lam > 1e-8:
                    f_w = float(getattr(config, "F_W", 0.595))
                    pred_depths_np = geom_piecewise_planar_inverse_depth_smoothing(
                        pred_depths_np,
                        f_w=f_w,
                        lam=lam,
                        rho=float(getattr(config, "smooth_rho", 1.0) or 1.0),
                        n_iter=int(getattr(config, "smooth_iter", 50) or 50),
                    )

                return get_ray_from_depth(pred_depths_np, V=config.V, F_W=config.F_W), pred_depths_np, None
            else:
                ray_logits, room_logits, _ = model(img_torch, mask_torch)
                ray_prob = F.softmax(ray_logits, dim=-1).squeeze(dim=0)
                sampled_indices = torch.argmax(ray_prob, dim=1)
                pred_semantics_np = sampled_indices.cpu().numpy()
                return get_ray_from_semantics(pred_semantics_np), pred_semantics_np, room_logits
    else:
        if "depth" in gt_data:
            pred_depths_np = gt_data["depth"]
            return get_ray_from_depth(pred_depths_np, V=config.V, F_W=config.F_W), pred_depths_np, None
        else:
            pred_semantics_np = gt_data["semantics"]
            return get_ray_from_semantics(pred_semantics_np), pred_semantics_np, None


def evaluate_predicted_weights(
    config,
    checkpoint_path: str,
    split_name: str,
    save_weights_json: str = "",
    print_weights: bool = False,
    print_every: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = float(getattr(config, "smooth_lambda", 0.0) or 0.0)
    print(f"Running IoU predicted-weights + geom-invdepth-smoothing(lam={lam}) on: {device}")

    split_file = os.path.join(config.dataset_dir, "processed", "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    is_zind = config.get("is_zind", False)
    if split_name not in ["train", "val", "test"]:
        raise ValueError(f"split must be one of ['train','val','test'], got: {split_name}")

    scene_list = split[split_name][: config.num_of_scenes]
    DatasetClass = LocalizationDatasetZind if is_zind else LocalizationDataset
    dataset = (
        DatasetClass(os.path.join(config.dataset_dir), scene_list)
        if is_zind
        else DatasetClass(os.path.join(config.dataset_dir, "processed"), scene_list)
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    img_h = int(ckpt.get("img_h", 320))
    img_w = int(ckpt.get("img_w", 640))
    map_h = int(ckpt.get("map_h", 256))
    map_w = int(ckpt.get("map_w", 256))
    num_weights = int(ckpt.get("num_weights", 3))

    if num_weights != 3:
        raise ValueError(f"This evaluator expects num_weights=3, got: {num_weights}")

    weight_model = WeightClassifierTwoStream(pretrained=False, num_weights=num_weights).to(device)
    weight_model.load_state_dict(ckpt["model_state"], strict=True)
    weight_model.eval()

    depth_net = None if config.use_ground_truth_depth else depth_net_pl.load_from_checkpoint(config.depth_weights).to(device).eval()
    semantic_net = None if config.use_ground_truth_semantic else semantic_net_pl.load_from_checkpoint(
        config.semantic_weights,
        num_classes=config.num_classes,
        num_room_types=config.num_room_types,
    ).to(device).eval()

    depth_df, semantic_df, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        dataset,
        os.path.join(config.dataset_dir, "processed") if not is_zind else config.dataset_dir,
        os.path.join(config.dataset_dir, "df"),
    )

    V_num = config.get("V", 9)
    F_W_val = config.get("F_W", 0.595)

    errors = {"refine": {"trans": [], "rot": []}}
    weight_fallback_used = 0
    used = 0

    predicted_weights_db = {} if save_weights_json else None
    map_cache = {}

    for data_idx in tqdm.tqdm(range(len(dataset)), desc="Samples", dynamic_ncols=True):
        data = dataset[data_idx]
        scene_idx = np.sum(data_idx >= np.array(dataset.scene_start_idx)) - 1
        scene_name_raw = dataset.scene_names[scene_idx]
        scene_name = _normalize_scene_name(scene_name_raw, is_zind)

        if scene_name not in valid_scene_names:
            continue

        frame_idx = data_idx - dataset.scene_start_idx[scene_idx]
        if frame_idx < 0 or frame_idx >= len(gt_poses[scene_name]):
            continue

        gt_x, gt_y, gt_o = gt_poses[scene_name][frame_idx, :3]
        unique_id = f"{scene_name}_{frame_idx}"

        used_fallback = False
        try:
            img = _preprocess_query_img(data["ref_img"], img_h, img_w).to(device)
            map_tensor = _load_map_tensor_cached(dataset, scene_name_raw, map_h, map_w, map_cache).to(device)
            wc, wf = _predict_weights(weight_model, img, map_tensor)
        except Exception:
            wc, wf = WC_FALLBACK, WF_FALLBACK
            weight_fallback_used += 1
            used_fallback = True

        if print_weights and (print_every <= 1 or (used % print_every) == 0):
            tqdm.tqdm.write(
                f"[{unique_id}] wc={np.round(np.array(wc, dtype=np.float32), 4).tolist()} "
                f"wf={np.round(np.array(wf, dtype=np.float32), 4).tolist()} "
                f"fallback={used_fallback}"
            )

        if predicted_weights_db is not None:
            predicted_weights_db[unique_id] = {"wc": wc, "wf": wf}

        used += 1

        img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
        mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)

        pred_rays_depth, pred_depths_np, _ = get_predicted_rays_with_smoothing(
            depth_net,
            img_torch,
            mask_torch,
            config,
            config.use_ground_truth_depth,
            {"depth": data["ref_depth"]},
        )
        pred_rays_semantic, pred_semantics_np, room_logits = get_predicted_rays_with_smoothing(
            semantic_net,
            img_torch,
            mask_torch,
            config,
            config.use_ground_truth_semantic,
            {"semantics": data["ref_semantics"]},
        )

        prob_vol_depth, _, _, _ = localize(torch.tensor(depth_df[scene_name]), torch.tensor(pred_rays_depth), return_np=False)
        prob_vol_semantic, _, _, _ = localize(torch.tensor(semantic_df[scene_name]), torch.tensor(pred_rays_semantic), return_np=False)
        prob_vol_iou = localize_iou(torch.tensor(depth_df[scene_name]), torch.tensor(pred_rays_depth), return_np=False)

        room_polygons = []
        if config.use_room_aware and room_logits is not None:
            room_polygons = predict_room_and_get_polygons(
                room_logits,
                data["room_polygons"],
                config.room_selection_threshold,
                is_zind,
            )

        combined_prob_vol = combine_prob_volumes_3(
            prob_vol_depth,
            prob_vol_semantic,
            prob_vol_iou,
            wc[0], wc[1], wc[2],
        )

        _, prob_dist, orient_map, _ = finalize_localization(combined_prob_vol, data["room_polygons"], room_polygons)

        candidates = extract_top_k_locations(
            prob_dist,
            orient_map,
            K=config.top_k,
            min_dist_m=config.min_dist_m,
            resolution_m_per_pixel=config.resolution_m_per_pixel,
        )

        loc, orient, _ = refine_and_select_best_candidate_iou(
            candidates,
            walls[scene_name],
            maps[scene_name],
            pred_depths_np,
            pred_semantics_np,
            wf,
            V_num,
            F_W_val,
        )

        if loc:
            errors["refine"]["trans"].append(np.sqrt((loc[0] - gt_x) ** 2 + (loc[1] - gt_y) ** 2))
            errors["refine"]["rot"].append(angular_difference_deg(orient, gt_o))

    refine_recalls = calculate_recalls(np.array(errors["refine"]["trans"]), np.array(errors["refine"]["rot"]))

    results_dir = os.path.join(config.results_dir, "predicted_weights_iou_geomfused")
    os.makedirs(results_dir, exist_ok=True)

    smooth_tag = f"lam{lam}_rho{float(getattr(config, 'smooth_rho', 1.0) or 1.0)}_iter{int(getattr(config, 'smooth_iter', 50) or 50)}"
    out_path = os.path.join(results_dir, f"refine_recalls_predicted_weights_{split_name}_{smooth_tag}.json")

    with open(out_path, "w") as f:
        json.dump(
            {
                "split": split_name,
                "checkpoint": checkpoint_path,
                "recalls": refine_recalls,
                "used_samples": int(used),
                "weight_fallback_used": int(weight_fallback_used),
                "wc_fallback": WC_FALLBACK,
                "wf_fallback": WF_FALLBACK,
                "img_size": [img_h, img_w],
                "map_size": [map_h, map_w],
                "smooth_lambda": lam,
                "smooth_rho": float(getattr(config, "smooth_rho", 1.0) or 1.0),
                "smooth_iter": int(getattr(config, "smooth_iter", 50) or 50),
            },
            f,
            indent=2,
        )

    if predicted_weights_db is not None:
        os.makedirs(os.path.dirname(save_weights_json), exist_ok=True)
        with open(save_weights_json, "w") as f:
            json.dump(predicted_weights_db, f, indent=2)
        print(f"Saved predicted weights to: {save_weights_json}")

    print("Refine recalls (predicted weights + smoothing):", refine_recalls)
    print(f"Used samples: {used}, weight fallback used: {weight_fallback_used}")
    print(f"Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IoU using predicted weights + geom inverse-depth piecewise-planar fused-lasso smoothing."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval.yaml",
        help="Path to yaml config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/weight_mlp_iou_twostream_trainable_map/mlp_weight_predictor_twostream_best.pt",
        help="Path to trained weight predictor checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--save_weights_json",
        type=str,
        default="",
        help="Optional path to save per-sample predicted weights JSON (unique_id -> wc/wf).",
    )
    parser.add_argument(
        "--no_print_weights",
        action="store_true",
        help="Disable printing predicted wc/wf for each sample.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print weights every N used samples (default: 1).",
    )

    parser.add_argument(
        "--smooth_lambda",
        type=float,
        default=None,
        help="Geometric inverse-depth smoothing strength (0 disables smoothing). If omitted, uses config value (or 0).",
    )
    parser.add_argument(
        "--smooth_rho",
        type=float,
        default=None,
        help="ADMM rho parameter for smoothing. If omitted, uses config value (or 1).",
    )
    parser.add_argument(
        "--smooth_iter",
        type=int,
        default=None,
        help="ADMM iterations for smoothing. If omitted, uses config value (or 50).",
    )

    args = parser.parse_args()

    config_path = _resolve_config_path(args.config_file)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)

    with open(config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    if args.smooth_lambda is not None:
        config.smooth_lambda = float(args.smooth_lambda)
    if args.smooth_rho is not None:
        config.smooth_rho = float(args.smooth_rho)
    if args.smooth_iter is not None:
        config.smooth_iter = int(args.smooth_iter)

    seed_everything(42)
    evaluate_predicted_weights(
        config,
        checkpoint_path,
        args.split,
        args.save_weights_json,
        (not args.no_print_weights),
        args.print_every,
    )


if __name__ == "__main__":
    main()
