import os
import argparse
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from attrdict import AttrDict
from PIL import Image

from data_utils.data_utils import LocalizationDataset
from data_utils.data_utils_zind import LocalizationDataset as LocalizationDatasetZind


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


class WeightLabelDatasetTwoStream(Dataset):
    def __init__(self, dataset, weights_db, img_size=(320, 640), map_size=(256, 256), num_weights=2):
        self.dataset = dataset
        self.weights_db = weights_db
        self.img_h, self.img_w = img_size
        self.map_h, self.map_w = map_size
        self.num_weights = int(num_weights)
        self.indices = []
        
        # ImageNet stats for perspective images
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Map stats (simple 0.5 mean/std for now, as maps are generated logic)
        self.map_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.map_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

        # Default weights for comparison / fallback
        # 2D: [depth, semantic]
        # 3D: [depth, semantic, iou]
        if self.num_weights == 2:
            self.default_wc = np.array([0.6, 0.4], dtype=np.float32)
            self.default_wf = np.array([0.3, 0.7], dtype=np.float32)
        elif self.num_weights == 3:
            self.default_wc = np.array([0.55, 0.40, 0.05], dtype=np.float32)
            self.default_wf = np.array([0.20, 0.70, 0.10], dtype=np.float32)
        else:
            raise ValueError(f"num_weights must be 2 or 3, got: {self.num_weights}")

        for idx in range(len(self.dataset)):
            scene_idx = np.sum(idx >= np.array(self.dataset.scene_start_idx)) - 1
            scene_name = self.dataset.scene_names[scene_idx]
            if "floor" not in scene_name:
                scene_name = f"scene_{int(scene_name.split('_')[1])}"
            frame_idx = idx - self.dataset.scene_start_idx[scene_idx]
            unique_id = f"{scene_name}_{frame_idx}"
            if unique_id in self.weights_db:
                self.indices.append(idx)
        
        print(f"TwoStream Dataset: Loaded {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        data = self.dataset[idx]

        scene_idx = np.sum(idx >= np.array(self.dataset.scene_start_idx)) - 1
        scene_name = self.dataset.scene_names[scene_idx]
        if "floor" not in scene_name:
            scene_name_raw = scene_name
            scene_name = f"scene_{int(scene_name.split('_')[1])}"
        else:
            scene_name_raw = scene_name
            
        frame_idx = idx - self.dataset.scene_start_idx[scene_idx]
        unique_id = f"{scene_name}_{frame_idx}"

        # 1. Load Perspective Image
        img = torch.tensor(data["ref_img"], dtype=torch.float32)  # CHW, 0-1
        if img.shape[1] != self.img_h or img.shape[2] != self.img_w:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.img_h, self.img_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        img = (img - self.img_mean) / self.img_std

        # 2. Load Semantic Map
        
        map_path = None
        
    
        base_dir = self.dataset.data_dir # LocalizationDataset usually stores it as data_dir
        
        potential_path = os.path.join(base_dir, scene_name_raw, "floorplan_semantic.png")
        
        if not os.path.exists(potential_path):
             # Try falling back to raw structure if processed path fails
             potential_path = os.path.join(base_dir, "..", "raw_S3D_perspective", scene_name_raw, "floorplan_semantic.png")


        if os.path.exists(potential_path):
            try:
                map_img = Image.open(potential_path).convert("RGB")
                map_tensor = transforms.ToTensor()(map_img) # CHW, 0-1
                # Resize map to fixed size (e.g. 256x256)
                map_tensor = F.interpolate(
                    map_tensor.unsqueeze(0), size=(self.map_h, self.map_w), mode="nearest"
                ).squeeze(0)
                map_tensor = (map_tensor - self.map_mean) / self.map_std
            except Exception as e:
                # Fallback if map load fails (should not happen if data is complete)
                # raise error instead of black map
                raise RuntimeError(f"Error loading map {potential_path}: {e}")
        else:
             # Map not found
             raise FileNotFoundError(f"Map not found at potential locations for {scene_name_raw}. Checked: {potential_path}")


        # 3. Load Targets & Weights
        weights_info = self.weights_db[unique_id]

        wc = np.array(weights_info.get("wc", self.default_wc.tolist()), dtype=np.float32)
        wf = np.array(weights_info.get("wf", self.default_wf.tolist()), dtype=np.float32)

        # Pad/truncate to expected dimension
        if wc.shape[0] < self.num_weights:
            wc = np.pad(wc, (0, self.num_weights - wc.shape[0]), mode="constant", constant_values=0.0)
        if wf.shape[0] < self.num_weights:
            wf = np.pad(wf, (0, self.num_weights - wf.shape[0]), mode="constant", constant_values=0.0)
        wc = wc[: self.num_weights]
        wf = wf[: self.num_weights]

        # Normalize to probability simplex for KLDivLoss
        wc = np.clip(wc, 0.0, None)
        wf = np.clip(wf, 0.0, None)
        wc_sum = float(np.sum(wc))
        wf_sum = float(np.sum(wf))
        wc = wc / wc_sum if wc_sum > 0 else self.default_wc.copy()
        wf = wf / wf_sum if wf_sum > 0 else self.default_wf.copy()

        # Calculate deviation weight (Hard Mining logic)
        
        gamma = 10.0
        dist_wc = float(np.sum(np.abs(wc - self.default_wc)))
        dist_wf = float(np.sum(np.abs(wf - self.default_wf)))
        alpha_c = 1.0 + gamma * dist_wc
        alpha_f = 1.0 + gamma * dist_wf

        target_wc = torch.tensor(wc, dtype=torch.float32)
        target_wf = torch.tensor(wf, dtype=torch.float32)
        # Keep the dataloader return signature unchanged (5 items):
        # provide a 2-vector weight per sample: [alpha_c, alpha_f]
        sample_weight = torch.tensor([alpha_c, alpha_f], dtype=torch.float32)

        return img, map_tensor, target_wc, target_wf, sample_weight


class WeightClassifierTwoStream(nn.Module):
    def __init__(self, hidden=512, dropout=0.2, pretrained=False, num_weights=2):
        super().__init__()

        self.num_weights = int(num_weights)
        if self.num_weights not in (2, 3):
            raise ValueError(f"num_weights must be 2 or 3, got: {self.num_weights}")
        
        # Stream 1: Image (ResNet50)
        weights_r50 = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.img_backbone = models.resnet50(weights=weights_r50)
        self.img_backbone.fc = nn.Identity() # Output: 2048
        
        # Stream 2: Map (ResNet18 - lighter)
        weights_r18 = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.map_backbone = models.resnet18(weights=weights_r18)
        self.map_backbone.fc = nn.Identity() # Output: 512
        
        # Fusion & Prediction
        fusion_dim = 2048 + 512
        self.fc1 = nn.Linear(fusion_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        
        self.fc_wc = nn.Linear(hidden, self.num_weights)
        self.fc_wf = nn.Linear(hidden, self.num_weights)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, img, map_img):
        # Image Stream
        img_feat = self.img_backbone(img) # (B, 2048)
        
        # Map Stream
        map_feat = self.map_backbone(map_img) # (B, 512)
        
        # Fusion
        x = torch.cat([img_feat, map_feat], dim=1) # (B, 2560)
        
        # MLP Head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        wc_logits = self.fc_wc(x)
        wf_logits = self.fc_wf(x)
        
        return wc_logits, wf_logits


def split_indices(n, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def train(
    config,
    weights_file,
    output_dir,
    img_h=320,
    img_w=640,
    map_h=256,
    map_w=256,
    batch_size=4,
    epochs=5,
    lr=1e-4,
    val_ratio=0.1,
    pretrained=False,
    amp=False,
    freeze_backbone=False,
    freeze_img_backbone=None,
    freeze_map_backbone=None,
    num_weights=2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_file = os.path.join(config.dataset_dir, "processed", "split.yaml")
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))

    is_zind = config.get("is_zind", False)
    DatasetClass = LocalizationDatasetZind if is_zind else LocalizationDataset
    dataset = DatasetClass(os.path.join(config.dataset_dir), split.train) if is_zind else DatasetClass(
        os.path.join(config.dataset_dir, "processed"), split.train
    )

    with open(weights_file, "r") as f:
        weights_db = json.load(f)

    # Use TwoStream Dataset
    full_ds = WeightLabelDatasetTwoStream(
        dataset,
        weights_db,
        img_size=(img_h, img_w),
        map_size=(map_h, map_w),
        num_weights=num_weights,
    )

    train_idx, val_idx = split_indices(len(full_ds), val_ratio=val_ratio, seed=42)
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(
        f"TwoStream Split: train={len(train_ds)} val={len(val_ds)} "
        f"| batches(train)={len(train_loader)} batches(val)={len(val_loader)} "
        f"| batch_size={batch_size}"
    )

    model = WeightClassifierTwoStream(pretrained=pretrained, num_weights=num_weights).to(device)

    if freeze_img_backbone is None:
        freeze_img_backbone = freeze_backbone
    if freeze_map_backbone is None:
        freeze_map_backbone = freeze_backbone

    if freeze_img_backbone:
        for param in model.img_backbone.parameters():
            param.requires_grad = False
    if freeze_map_backbone:
        for param in model.map_backbone.parameters():
            param.requires_grad = False

    def _count_params(module: nn.Module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    total_all, trainable_all = _count_params(model)
    total_img, trainable_img = _count_params(model.img_backbone)
    total_map, trainable_map = _count_params(model.map_backbone)
    print(
        "Trainability: "
        f"freeze_img_backbone={bool(freeze_img_backbone)} "
        f"freeze_map_backbone={bool(freeze_map_backbone)}\n"
        f"  model: trainable {trainable_all}/{total_all} params\n"
        f"  img_backbone: trainable {trainable_img}/{total_img} params\n"
        f"  map_backbone: trainable {trainable_map}/{total_map} params"
    )
    try:
        print(f"map_backbone.conv1.weight.requires_grad = {model.map_backbone.conv1.weight.requires_grad}")
    except Exception:
        pass

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.KLDivLoss(reduction="none") 
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    os.makedirs(output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        # Check input unpacking: now we have 5 items
        for img, map_img, target_wc, target_wf, weights in tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False):
            img = img.to(device)
            map_img = map_img.to(device)
            target_wc = target_wc.to(device)
            target_wf = target_wf.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                # Forward pass with two inputs
                wc_logits, wf_logits = model(img, map_img)
                
                wc_log_prob = F.log_softmax(wc_logits, dim=1)
                wf_log_prob = F.log_softmax(wf_logits, dim=1)
                
                loss_wc_raw = criterion(wc_log_prob, target_wc).sum(dim=1)
                loss_wf_raw = criterion(wf_log_prob, target_wf).sum(dim=1)

                # weights: (B, 2) -> [:,0] for coarse, [:,1] for fine
                if weights.ndim == 2 and weights.shape[1] == 2:
                    w_c = weights[:, 0]
                    w_f = weights[:, 1]
                else:
                    # Backward-compatible fallback: single scalar weight
                    w_c = weights
                    w_f = weights

                loss = (loss_wc_raw * w_c + loss_wf_raw * w_f).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * img.size(0)

        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, map_img, target_wc, target_wf, weights in tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False):
                img = img.to(device)
                map_img = map_img.to(device)
                target_wc = target_wc.to(device)
                target_wf = target_wf.to(device)
                weights = weights.to(device)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    wc_logits, wf_logits = model(img, map_img)
                    
                    wc_log_prob = F.log_softmax(wc_logits, dim=1)
                    wf_log_prob = F.log_softmax(wf_logits, dim=1)
                    
                    loss_wc_raw = criterion(wc_log_prob, target_wc).sum(dim=1)
                    loss_wf_raw = criterion(wf_log_prob, target_wf).sum(dim=1)

                    if weights.ndim == 2 and weights.shape[1] == 2:
                        w_c = weights[:, 0]
                        w_f = weights[:, 1]
                    else:
                        w_c = weights
                        w_f = weights

                    loss = (loss_wc_raw * w_c + loss_wf_raw * w_f).mean()
                    
                val_loss += loss.item() * img.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        print(f"Epoch {epoch:02d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(output_dir, "mlp_weight_predictor_twostream_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "img_h": img_h,
                    "img_w": img_w,
                    "map_h": map_h,
                    "map_w": map_w,
                    "num_weights": int(num_weights),
                    "freeze_img_backbone": bool(freeze_img_backbone),
                    "freeze_map_backbone": bool(freeze_map_backbone),
                    "mode": "twostream_kldiv_hard",
                },
                ckpt_path,
            )

    final_path = os.path.join(output_dir, "mlp_weight_predictor_twostream_last.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "img_h": img_h,
            "img_w": img_w,
            "map_h": map_h,
            "map_w": map_w,
            "num_weights": int(num_weights),
            "freeze_img_backbone": bool(freeze_img_backbone),
            "freeze_map_backbone": bool(freeze_map_backbone),
            "mode": "twostream_kldiv_hard",
        },
        final_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Two-Stream MLP (Image+Map) with weighted loss.")
    parser.add_argument("--config_file", type=str, default="evaluation/configuration/S3D/config_eval.yaml")
    parser.add_argument("--weights_file", type=str, default="Data/S3D/best_weights_labels.json")
    parser.add_argument("--output_dir", type=str, default="results/weight_mlp")
    parser.add_argument("--img_h", type=int, default=320)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--map_h", type=int, default=256)
    parser.add_argument("--map_w", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet50")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ResNet backbone to save memory")
    parser.add_argument("--num_weights", type=int, default=3, choices=[2, 3], help="Output weight dimension: 2=[d,s], 3=[d,s,iou]")
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
        freeze_backbone=args.freeze_backbone,
        num_weights=args.num_weights,
    )


if __name__ == "__main__":
    main()
