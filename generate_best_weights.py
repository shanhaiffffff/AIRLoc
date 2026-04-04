
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch
import yaml
from attrdict import AttrDict
import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.depth.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from data_utils.data_utils import LocalizationDataset
from data_utils.data_utils_zind import LocalizationDataset as LocalizationDatasetZind
from utils.data_loader_helper import load_scene_data
from utils.localization_utils import localize, finalize_localization
from evaluation.candidate_extractor import extract_top_k_locations
from evaluation.room_predictor import predict_room_and_get_polygons
from evaluation.eval_localization_iou import (
    seed_everything,
    localize_iou,
    combine_prob_volumes_3,
    get_predicted_rays,
    refine_and_select_best_candidate_iou,
    angular_difference_deg,
)


WC_DEFAULT = [0.55, 0.40, 0.05]
WF_DEFAULT = [0.20, 0.70, 0.10]

WC_CANDIDATES = [
    [0.55, 0.40, 0.05], [0.6, 0.35, 0.05], [0.65, 0.30, 0.05], [0.7, 0.25, 0.05],
    [0.5, 0.45, 0.05], [0.50, 0.40, 0.10], [0.55, 0.35, 0.10], [0.60, 0.30, 0.10],
]

WF_CANDIDATES = [
    [0.20, 0.70, 0.10], [0.25, 0.65, 0.10], [0.30, 0.60, 0.10],
    [0.25, 0.70, 0.05], [0.30, 0.65, 0.05],
]


def _pair_distance_to_default(wc, wf):
    wc_dist = sum(abs(a - b) for a, b in zip(wc, WC_DEFAULT))
    wf_dist = sum(abs(a - b) for a, b in zip(wf, WF_DEFAULT))
    return wc_dist + wf_dist


def _sample_score(loc, orient, gt_x, gt_y, gt_o):
    score = 0
    if loc:
        err_trans = np.sqrt((loc[0] - gt_x) ** 2 + (loc[1] - gt_y) ** 2)
        err_rot = angular_difference_deg(orient, gt_o)

        if err_trans <= 1.0:
            score += 1
        if err_trans <= 0.5:
            score += 1
        if err_trans <= 0.1:
            score += 1
        if err_trans <= 1.0 and err_rot <= 30.0:
            score += 1
    return score


def generate_best_weights(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("Searching per-sample best IoU weights (coarse + fine)")
    print(f"Default tie-break pair: wc={WC_DEFAULT}, wf={WF_DEFAULT}")
    print(f"#coarse={len(WC_CANDIDATES)}, #fine={len(WF_CANDIDATES)}, #combos={len(WC_CANDIDATES) * len(WF_CANDIDATES)}")
    print("=" * 72)

    split_file = os.path.join(config.dataset_dir, "processed", "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    is_zind = config.get("is_zind", False)
    split_name = "train"
    max_scenes = config.get("num_of_scenes", 30000)

    scene_list = split[split_name][:max_scenes]
    DatasetClass = LocalizationDatasetZind if is_zind else LocalizationDataset
    dataset = DatasetClass(os.path.join(config.dataset_dir), scene_list) if is_zind else DatasetClass(
        os.path.join(config.dataset_dir, "processed"), scene_list
    )

    depth_net = None if config.use_ground_truth_depth else depth_net_pl.load_from_checkpoint(config.depth_weights).to(device).eval()
    semantic_net = None if config.use_ground_truth_semantic else semantic_net_pl.load_from_checkpoint(
        config.semantic_weights,
        num_classes=config.num_classes,
        num_room_types=config.num_room_types,
    ).to(device).eval()

    print("Pre-loading maps...")
    depth_df, semantic_df, maps, gt_poses, valid_scene_names, walls = load_scene_data(
        dataset,
        os.path.join(config.dataset_dir, "processed") if not is_zind else config.dataset_dir,
        os.path.join(config.dataset_dir, "df"),
    )

    combos = [(wc, wf) for wc, wf in itertools.product(WC_CANDIDATES, WF_CANDIDATES)]
    best_weights_db = {}

    V_num = config.get("V", 9)
    F_W_val = config.get("F_W", 0.595)

    pbar = tqdm.tqdm(range(len(dataset)), dynamic_ncols=True, desc="Samples")
    for data_idx in pbar:
        data = dataset[data_idx]
        scene_idx = np.sum(data_idx >= np.array(dataset.scene_start_idx)) - 1
        scene_name = dataset.scene_names[scene_idx]
        if not is_zind and "floor" not in scene_name:
            scene_name = f"scene_{int(scene_name.split('_')[1])}"

        if scene_name not in valid_scene_names:
            continue

        frame_idx = data_idx - dataset.scene_start_idx[scene_idx]
        if frame_idx < 0 or frame_idx >= len(gt_poses[scene_name]):
            continue

        gt_x, gt_y, gt_o = gt_poses[scene_name][frame_idx, :3]
        unique_id = f"{scene_name}_{frame_idx}"

        img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)
        mask_torch = torch.tensor(data["ref_mask"], device=device).unsqueeze(0)

        pred_rays_depth, pred_depths_np, _ = get_predicted_rays(
            depth_net,
            img_torch,
            mask_torch,
            config,
            config.use_ground_truth_depth,
            {"depth": data["ref_depth"]},
        )
        pred_rays_semantic, pred_semantics_np, room_logits = get_predicted_rays(
            semantic_net,
            img_torch,
            mask_torch,
            config,
            config.use_ground_truth_semantic,
            {"semantics": data["ref_semantics"]},
        )

        prob_vol_depth, _, _, _ = localize(
            torch.tensor(depth_df[scene_name]),
            torch.tensor(pred_rays_depth),
            return_np=False,
        )
        prob_vol_semantic, _, _, _ = localize(
            torch.tensor(semantic_df[scene_name]),
            torch.tensor(pred_rays_semantic),
            return_np=False,
        )
        prob_vol_iou = localize_iou(
            torch.tensor(depth_df[scene_name]),
            torch.tensor(pred_rays_depth),
            return_np=False,
        )

        room_polygons = []
        if config.use_room_aware and room_logits is not None:
            room_polygons = predict_room_and_get_polygons(
                room_logits,
                data["room_polygons"],
                config.room_selection_threshold,
                is_zind,
            )

        # Cache coarse candidates for each wc on this sample.
        coarse_candidates_cache = {}
        for wc in WC_CANDIDATES:
            combined_prob_vol = combine_prob_volumes_3(
                prob_vol_depth,
                prob_vol_semantic,
                prob_vol_iou,
                wc[0], wc[1], wc[2],
            )
            _, prob_dist, orient_map, _ = finalize_localization(
                combined_prob_vol,
                data["room_polygons"],
                room_polygons,
            )
            candidates = extract_top_k_locations(
                prob_dist,
                orient_map,
                K=config.top_k,
                min_dist_m=config.min_dist_m,
                resolution_m_per_pixel=config.resolution_m_per_pixel,
            )
            coarse_candidates_cache[str(wc)] = candidates

        # Find best combo for this sample.
        best_score = -1
        best_err_trans = float("inf")
        best_dist = float("inf")
        best_wc = WC_DEFAULT
        best_wf = WF_DEFAULT

        eps = 1e-9

        for wc, wf in combos:
            candidates = coarse_candidates_cache[str(wc)]
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

            score = _sample_score(loc, orient, gt_x, gt_y, gt_o)
            if loc:
                err_trans = float(np.sqrt((loc[0] - gt_x) ** 2 + (loc[1] - gt_y) ** 2))
            else:
                err_trans = float("inf")
            dist = _pair_distance_to_default(wc, wf)

            if score > best_score:
                best_score = score
                best_err_trans = err_trans
                best_dist = dist
                best_wc = wc
                best_wf = wf
            elif score == best_score:
                # Tie-break 1: smaller translation error wins
                if err_trans < best_err_trans - eps:
                    best_err_trans = err_trans
                    best_dist = dist
                    best_wc = wc
                    best_wf = wf
                # Tie-break 2: if err_trans also ties, prefer default weights
                # (implemented as closest distance to default pair; default has distance 0)
                elif abs(err_trans - best_err_trans) <= eps and dist < best_dist - eps:
                    best_dist = dist
                    best_wc = wc
                    best_wf = wf

        best_weights_db[unique_id] = {
            'wc': [float(x) for x in best_wc],
            'wf': [float(x) for x in best_wf],
            'score': int(best_score)
        }

        pbar.set_postfix_str(f"Best: C={best_wc} F={best_wf} Score={best_score}")

    save_path = os.path.join(config.dataset_dir, "best_weights_labels.json")
    print(f"Saving to {save_path}...")
    with open(save_path, "w") as f:
        json.dump(best_weights_db, f, indent=2)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Search per-sample best coarse/fine weights.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="evaluation/configuration/S3D/config_eval.yaml",
        help="Path to yaml config file.",
    )
    args = parser.parse_args()

    config_path = args.config_file
    if not os.path.exists(config_path):
        fallback_candidates = [
            os.path.join("evaluation", "configuration", "S3D", args.config_file),
            os.path.join("evaluation", "configuration", args.config_file),
        ]
        for cand in fallback_candidates:
            if os.path.exists(cand):
                config_path = cand
                break

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {args.config_file}. "
            f"Tried: {args.config_file}, "
            f"evaluation/configuration/S3D/{args.config_file}, "
            f"evaluation/configuration/{args.config_file}"
        )

    with open(config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    seed_everything(42)
    generate_best_weights(config)


if __name__ == "__main__":
    main()
