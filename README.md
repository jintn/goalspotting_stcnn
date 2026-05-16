# Detecting Goals in Soccer Broadcasts Using Spatiotemporal Convolutional Neural Networks

Source code for the MSc thesis by Jin Tony Nymann (Oslo Metropolitan University, Spring 2026).

The project fine-tunes three Kinetics-400 pretrained spatiotemporal CNNs — **R(2+1)D-18**, **R3D-18**, and **MC3-18** — on the SoccerNet dataset to detect goal events in full-length broadcast halves. Predictions are produced with sliding-window inference and temporal post-processing (smoothing, thresholding, non-maximum suppression).

## Headline results

Test-set performance for the F1-optimised operating point on the SoccerNet test split (goal class, δ = 5 s):

| Model        | Precision | Recall | F1    | Tight Avg-AP (1–5 s) | Loose Avg-AP (5–60 s) | FP / half |
|--------------|-----------|--------|-------|----------------------|------------------------|-----------|
| R(2+1)D-18   | 0.119     | 0.911  | 0.210 | 0.292                | 0.318                  | 11.4      |
| R3D-18       | 0.095     | 0.905  | 0.172 | 0.237                | 0.269                  | 14.5      |
| MC3-18       | 0.106     | 0.902  | 0.190 | 0.252                | 0.288                  | 12.8      |

See thesis Tables 4.3 and 4.13 for the full comparison.

## Repository layout

```
.
├── 01_train.ipynb               # all 13 thesis training runs as named presets
├── 02_validation_sweep.ipynb    # sliding-window inference on val + post-processing sweep
├── 03_testset_evaluation.ipynb  # held-out test-set metrics at chosen operating points
├── scripts/
│   ├── build_jitter_cache.py    # standalone clip-cache builder with ±2s jitter
│   ├── benchmark_goal_inference.py
│   ├── download_val.py          # SoccerNet validation split download helper
│   ├── data_check.py            # cross-check SoccerNet labels against external sources
│   ├── dataset_table.py         # class-distribution summary
│   └── stage2_goal_verifier.py
└── README.md
```

## Setup

```bash
git clone https://github.com/jintn/r2plus1d_18_goalspotting.git
cd r2plus1d_18_goalspotting
pip install torch torchvision opencv-python numpy pandas scikit-learn tqdm jupyter \
            SoccerNet wandb seaborn matplotlib
```

SoccerNet videos and labels are not redistributed here. Register at <https://www.soccer-net.org/data> to receive an NDA password, then export it:

```bash
export NV_PASSWORD=<your_soccernet_password>
```

The first cell of each notebook automatically downloads any missing split (train, valid, or test) into `dataset/`. The downloader is idempotent — re-running it skips files already on disk.

Optional: set `WANDB_API_KEY` and `WANDB_PROJECT` to enable training-run logging. Without these the training loop runs locally without W&B.

## Reproducing the thesis experiments

All training runs live in `01_train.ipynb` as a registry. To reproduce any thesis row, set `EXPERIMENT` in the registry cell to the matching preset and run the notebook top-down:

| Preset                                         | Thesis section | Backbone     | Notes                                             |
|------------------------------------------------|----------------|--------------|---------------------------------------------------|
| `exp1_random_negs`                             | §4.1           | R(2+1)D-18   | binary, random negatives only                     |
| `exp1_hard_negs`                               | §4.1           | R(2+1)D-18   | binary, shots added as hard negatives             |
| `exp2_run1_dropout`                            | §4.2 Table 4.5 | R(2+1)D-18   | dropout                                           |
| `exp2_run2_dropout_freeze`                     | §4.2 Table 4.5 | R(2+1)D-18   | + layer freezing                                  |
| `exp2_run3_dropout_freeze_aug`                 | §4.2 Table 4.5 | R(2+1)D-18   | + augmentation                                    |
| `exp2_run4_aug`                                | §4.2 Table 4.5 | R(2+1)D-18   | augmentation only, with jitter                    |
| `exp2_run5_aug_no_jitter`                      | §4.2 Table 4.5 | R(2+1)D-18   | augmentation, no jitter — **best val F1 = 0.868** |
| `exp3_r2plus1d_1shot` / `exp3_r2plus1d_2shot`  | §4.3           | R(2+1)D-18   | 4-class, 1 vs 2 shots per goal                    |
| `exp3_r3d_1shot` / `exp3_r3d_2shot`            | §4.3           | R3D-18       | 4-class, 1 vs 2 shots per goal                    |
| `exp3_mc3_1shot` / `exp3_mc3_2shot`            | §4.3           | MC3-18       | 4-class, 1 vs 2 shots per goal                    |

Hyperparameters fixed across all runs (thesis Table 3.3): Adam, lr = 1×10⁻⁴, batch size 64, 4 s clips × 16 frames × 112², FP16 AMP. 8 epochs for Exp 1, 10 epochs for Exp 2 and 3.

### Evaluation flow

```text
01_train.ipynb                  → checkpoints/<experiment>/<timestamp>/best.pt
02_validation_sweep.ipynb       → results/validation_sweeps/<experiment>/{sweep_full,top5_by_f1,top5_by_recall}.csv
03_testset_evaluation.ipynb     → results/test_predictions/<experiment>/{headline_metrics,ap_vs_tolerance}.csv
```

Inference is decoupled from post-processing: `02_` and `03_` pickle the raw per-window confidence curves first, then sweep threshold / NMS / smoothing parameters in memory. Re-running a sweep after the pickle exists is cheap and requires no GPU.

## Hardware

All thesis numbers were produced on a single NVIDIA T4 GPU. The unified training notebook runs unchanged on consumer GPUs — drop `batch_size` in the registry if VRAM is the constraint; it does not affect reported metrics.

## Citation

```bibtex
@mastersthesis{nymann2026goalspotting,
  author  = {Nymann, Jin Tony},
  title   = {Detecting Goals in Soccer Broadcasts Using Spatiotemporal Convolutional Neural Networks},
  school  = {Oslo Metropolitan University},
  year    = {2026},
  type    = {Master's thesis}
}
```

## Acknowledgements

- Thesis supervisors: Pål Halvorsen and Mehdi Houshmand Sarkhoosh.
- SoccerNet dataset by Giancola et al. (<https://www.soccer-net.org>) — use is subject to the SoccerNet NDA.
- Backbone implementations from `torchvision.models.video` (R2Plus1D, R3D, MC3), pretrained on Kinetics-400.
