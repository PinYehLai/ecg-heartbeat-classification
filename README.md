# ECG Heartbeat Classification under Distribution Shift

Deep learning for **5-class ECG heartbeat image classification** under severe class imbalance and simulated cross-device / cross-hospital distribution shift.

This project was developed as a three-person team project at **Cornell Tech**. My primary technical contributions were the **DenseNet-121 experiments** and the design and implementation of the final **imbalance-aware ensemble strategy**.

The project explores how transfer learning, ECG-specific preprocessing, augmentation, class-imbalance mitigation, test-time augmentation, and model ensembling affect generalization on a challenging shifted test distribution.

---

## Highlights

* **87,554** labeled ECG training images
* **21,892** unlabeled test images
* **5 heartbeat classes**
* More than **80% of training samples belong to the normal class**
* Evaluated with **Macro-F1** to emphasize performance across minority classes
* Best recorded DenseNet-121 result: **0.72812 public Macro-F1**
* Best standalone team model: **0.74714 public Macro-F1**
* Final rule-based ensemble: **0.76374 public Macro-F1**

---

## Problem

The goal is to classify ECG heartbeat images into five categories.

Two characteristics make the task particularly challenging:

### Severe Class Imbalance

The normal heartbeat class (`class 0`) accounts for more than 80% of the training data. Models trained without explicit consideration of this imbalance can achieve high overall accuracy while performing poorly on minority heartbeat classes.

For this reason, **Macro-F1** is used as the main evaluation metric.

### Distribution Shift

The test distribution was intentionally shifted to simulate ECG images originating from different hospitals, devices, or recording conditions.

As a result, strong validation performance does not necessarily translate directly to strong test-set performance. Robustness to changes in waveform appearance, image style, and noise is therefore a central focus of the project.

---

## My Contributions

My primary contributions to the project were:

* Developed and evaluated multiple **DenseNet-121** transfer-learning pipelines.
* Adapted DenseNet-121 for both **single-channel grayscale** and **RGB ECG inputs**.
* Designed ECG-specific preprocessing and augmentation strategies.
* Experimented with **Mixup** and class-weighted loss to address overfitting and class imbalance.
* Implemented **5-view test-time augmentation (TTA)** for more stable inference under distribution shift.
* Designed and implemented the final **imbalance-aware rule-based ensemble** combining DenseNet-121, ResNet-18 + Attention, and EfficientNet predictions.
* Analyzed the models' shared tendency to over-predict the dominant normal class and designed the ensemble specifically around this failure mode.

The other standalone architectures were developed independently by my teammates.

---

## DenseNet-121 Experiments

I evaluated several DenseNet-121 configurations to understand how preprocessing, robustness techniques, and input representation affect performance.

| Experiment                    | Main Configuration                                                     | Public Macro-F1 |
| ----------------------------- | ---------------------------------------------------------------------- | --------------: |
| Simplified Grayscale DenseNet | 256×256 grayscale input, standard cross-entropy, single-view inference |     **0.70882** |
| Robust Grayscale DenseNet     | ECG-specific augmentation, Mixup, class-weighted loss, 5-view TTA      |     **0.72146** |
| RGB DenseNet                  | 256×256 RGB input with ImageNet normalization                          |     **0.72812** |

These experiments are preserved in:

[`notebooks/densenet121_experiments.ipynb`](notebooks/densenet121_experiments.ipynb)

### ECG-Specific Preprocessing

One robustness experiment uses a custom waveform preprocessing transform that:

1. converts the grayscale representation into an inverted waveform image, and
2. applies a max-pooling operation as a simple dilation-like transformation to emphasize thin ECG traces.

The objective is to make waveform structures more prominent while reducing sensitivity to differences in line thickness and background appearance.

### Mixup and Class Weighting

The robustness-focused DenseNet experiment also combines:

* **Mixup** to reduce model overconfidence and improve generalization, and
* smoothed inverse-frequency class weights to increase the influence of minority heartbeat classes without assigning excessively large weights to rare examples.

### Test-Time Augmentation

For the TTA experiment, each test image is evaluated across **five stochastic views** using small affine perturbations.

Class probabilities are aggregated across these views before generating the final prediction, reducing prediction variance under small input changes.

---

## Imbalance-Aware Ensemble

The three independently developed team models exhibited a similar error pattern: because the training set is dominated by class `0`, they tended to make conservative predictions and over-predict the normal class.

A conventional majority-voting ensemble can preserve this bias.

For example:

```text
EfficientNet   → abnormal
ResNet         → normal
DenseNet       → normal
```

Standard majority voting would classify the sample as normal even though one model detected an abnormal heartbeat.

To address this failure mode, we designed an **asymmetric rule-based ensemble** that prioritizes abnormal detection.

### Decision Rules

For each sample:

1. If all three models predict `0`, classify the sample as **normal**.
2. If at least one model predicts an abnormal class (`1–4`), ignore normal votes.
3. If only one abnormal prediction exists, use that prediction.
4. If multiple models agree on an abnormal class, use the abnormal majority.
5. If abnormal classes conflict, resolve the tie according to standalone model performance:

```text
EfficientNet
    ↓
ResNet-18 + Attention
    ↓
DenseNet-121
```

The complete implementation and recorded outputs are available in:

[`notebooks/ensemble_strategy.ipynb`](notebooks/ensemble_strategy.ipynb)

---

## Results

### Standalone Models and Ensemble

| Model                        | Public Macro-F1 |
| ---------------------------- | --------------: |
| DenseNet-121                 |     **0.72812** |
| ResNet-18 + Attention        |     **0.73530** |
| EfficientNet                 |     **0.74714** |
| **Imbalance-Aware Ensemble** |     **0.76374** |

The ensemble improved public Macro-F1 from **0.74714 to 0.76374** over the strongest standalone model.

### Ensemble Behavior

The recorded ensemble run produced:

* **3,901** samples classified as abnormal
* **3,387** abnormal predictions from the strongest standalone EfficientNet model
* **514 additional abnormal candidates** identified by the ensemble
* Only **43 cases** required model-ranking tie resolution

This suggests that most of the ensemble's behavior came from correcting the models' shared tendency toward false-normal predictions rather than repeatedly relying on the ranking rule.

---

## Key Findings

**1. Model capacity alone was not enough.**

CNN baselines trained from scratch saturated around a public Macro-F1 of approximately **0.67**, despite substantially stronger training performance.

**2. Distribution shift was a major challenge.**

DenseNet validation performance was much higher than public-test performance, highlighting a significant gap between in-domain validation and shifted test data.

**3. Robustness techniques helped, but their effects were not always additive.**

ECG-specific augmentation, Mixup, class weighting, TTA, and changes in input representation produced different trade-offs. More complex preprocessing did not automatically guarantee the highest leaderboard performance.

**4. Error-aware ensembling was more useful than generic majority voting.**

Designing the ensemble around a known failure mode—the tendency to over-predict the dominant normal class—improved Macro-F1 beyond every standalone model.

---

## Repository Structure

```text
ecg-heartbeat-classification/
│
├── notebooks/
│   ├── densenet121_experiments.ipynb
│   └── ensemble_strategy.ipynb
│
├── results/
│   └── model_results.csv
│
├── README.md
└── requirements.txt
```

### `densenet121_experiments.ipynb`

Contains the DenseNet-121 experimentation workflow, including:

* preprocessing
* augmentation
* transfer learning
* Mixup
* class weighting
* training and validation
* test-time augmentation
* recorded experiment results

### `ensemble_strategy.ipynb`

Contains:

* ensemble motivation
* model prediction alignment
* imbalance-aware decision rules
* abnormal-class conflict resolution
* recorded ensemble diagnostics and results

### `results/model_results.csv`

Provides a compact summary of the recorded public Macro-F1 results.

---

## Tech Stack

**Language**

* Python

**Deep Learning**

* PyTorch
* Torchvision
* DenseNet-121

**Data / Evaluation**

* NumPy
* Pandas
* scikit-learn

**Development**

* Jupyter Notebook
* Google Colab
* CUDA

**Techniques**

* Transfer Learning
* Computer Vision
* Data Augmentation
* Mixup
* Class-Weighted Loss
* Test-Time Augmentation
* Ensemble Learning
* Class-Imbalance Handling
* Distribution-Shift Analysis

---

## Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The original ECG dataset, trained model checkpoints, and teammate prediction files are **not included** in this repository.

The notebooks are therefore intended primarily to preserve and document the project's implementation, experiments, outputs, and design decisions rather than serve as a fully packaged reproduction environment.

---

## Limitations and Future Work

Several limitations remain:

* The ensemble uses a fixed model-priority rule based on standalone leaderboard performance.
* Probability outputs were not explicitly calibrated before ensembling.
* The dataset exhibits substantial train/test distribution shift.
* Computational constraints limited broader architecture and hyperparameter exploration.
* The current ensemble operates on discrete class predictions rather than calibrated uncertainty estimates.

Potential extensions include:

* confidence-aware or uncertainty-aware ensembling,
* probability calibration,
* adaptive class-specific decision thresholds,
* self-supervised pretraining on ECG images,
* stronger domain-generalization techniques, and
* evaluation across independently collected ECG datasets.

---

## Team and Attribution

This project was completed as a **three-person team project at Cornell Tech**.

Each team member developed an independent final model:

* **Alice** — EfficientNet-based approach
* **David** — ResNet-18 with spatial attention
* **Andrew (Pin-Yeh Lai)** — DenseNet-121 experiments

The final ensemble combined predictions from all three models.

My primary technical contributions were the **DenseNet-121 experimentation pipeline** and the **design and implementation of the imbalance-aware ensemble strategy** presented in this repository.

---

## Disclaimer

This repository documents an **academic machine-learning project** and is intended for educational and portfolio purposes only.

The models have **not been clinically validated** and should not be used for medical diagnosis or clinical decision-making.
