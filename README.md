# AIRLoc
This is the official Pytorch implementation of our paper: "AIRLoc: An IoU-Guided Adaptive Fusion Method for Floorplan-Based Visual Localization"

## Setup

We conducted experiments based on SemRayLoc. You can download the dataset from SemRayLoc and use its depth and semantics estimation method. 

[SemRayLoc](https://github.com/TAU-VAILab/SemRayLoc) 

## Usage

### Generate optimal weight pseudo-labels
```
python -m evaluation.generate_best_weights --config_file config_eval.yaml
```

### train mlp
You need to modify the checkpoint's saving path according to your needs.
```
python -m evaluation.train_mlp_weight --config_file config_eval.yaml --weights_file <weights_file> --output_dir <output_dir>
```

### evaluate

```
python -m evaluation.eval_localization --config_file config_eval.yaml --checkpoint <checkpoint> --split test --smooth_lambda 0.01 --smooth_iter 50
```

## Acknowledgement

We thank the authors of [SemRayLoc](https://github.com/TAU-VAILab/SemRayLoc) for releasing their helpful codebases.
