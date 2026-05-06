# ASL Gesture Recognition with CNNs and Transfer Learning

A PyTorch computer-vision project for classifying American Sign Language (ASL) hand gestures using a custom convolutional neural network and transfer learning with AlexNet feature extraction.

This repository converts a university lab into a GitHub-ready ML engineering project with clean code structure, reproducible training, model checkpoints, hyperparameter tuning, and held-out test evaluation.

## Why this project is useful

- Image classification with PyTorch and `torchvision.datasets.ImageFolder`
- Custom CNN architecture design for 9-class hand gesture recognition
- Training/validation/test splitting with balanced class counts
- Sanity checking by overfitting a small dataset
- Hyperparameter tuning across learning rate, batch size, and model capacity
- Transfer learning using pretrained AlexNet convolutional features
- Reproducible evaluation on a held-out test set
- Modular code that is easier to review than a notebook-only implementation

## Task

Given an image of a hand gesture, classify it as one of the ASL letters:

```text
A, B, C, D, E, F, G, H, I
```

## Dataset format

The project expects an ImageFolder-style dataset:

```text
data/asl_gestures/
├── A/
├── B/
├── C/
├── D/
├── E/
├── F/
├── G/
├── H/
└── I/
```

The lab used 2,219 total images and split them approximately as:

| Split | Images |
|---|---:|
| Train | 1,553 |
| Validation | 332 |
| Test | 334 |

## Repository structure

```text
asl-gesture-recognition-cnn-transfer-learning/
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── transfer_learning.py
│   ├── plot_curves.py
│   └── utils.py
├── experiments/
│   └── run_experiments.py
├── checkpoints/
├── results/
└── data/
```

## Setup

```bash
git clone <your-repo-url>
cd asl-gesture-recognition-cnn-transfer-learning
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Train the custom CNN

```bash
python -m src.train --data-dir data/asl_gestures --model custom --batch-size 32 --learning-rate 0.0003 --epochs 10
```

## Sanity-check by overfitting a tiny subset

```bash
python -m src.train --data-dir data/asl_gestures --model custom --batch-size 64 --learning-rate 0.001 --epochs 200 --small-overfit
```

## Evaluate a checkpoint

```bash
python -m src.evaluate --data-dir data/asl_gestures --checkpoint checkpoints/best_custom.pt --model custom
```

## Transfer learning workflow

Extract frozen AlexNet features:

```bash
python -m src.transfer_learning extract --data-dir data/asl_gestures --output-dir results/alexnet_features
```

Train a classifier head on those features:

```bash
python -m src.transfer_learning train --feature-dir results/alexnet_features --epochs 10 --batch-size 32 --learning-rate 0.0003
```

## Run hyperparameter experiments

```bash
python experiments/run_experiments.py --data-dir data/asl_gestures
```

## Representative lab results

| Approach | Validation accuracy | Test accuracy |
|---|---:|---:|
| Custom CNN | ~89.6% | ~79.9% |
| AlexNet feature transfer learning | ~95.8% | ~94.0% |

The transfer-learning model achieved a large improvement because pretrained AlexNet features already encode useful visual patterns such as edges, curves, textures, and shapes. The smaller classifier head then learns the ASL-specific mapping using far fewer trainable parameters.
