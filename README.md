# Deep Learning — Fine-Grained Bird Classification & Agatha Christie Text Generation

Coursework for **CSC8637 "Deep Learning"**, MSc Data Science & AI, Newcastle University (2026).

This repository contains two deep learning systems built and evaluated from end to end:

1. **Fine-grained image classification** — distinguishing 200 visually similar bird species.
2. **Autoregressive text generation** — a Transformer language model that writes in the style of Agatha Christie.

---

## Aim

Fine-grained classification asks a model to separate categories that differ only in subtle detail (a wing bar, a beak shape) rather than coarse object types. The aim of this project is twofold:

- **Task 1:** Train two image classifiers on the Caltech-UCSD Birds-200-2011 dataset — one built on a standard pretrained backbone via transfer learning, and one convolutional network designed from scratch — and quantify the gap between them.
- **Task 2:** Train a Transformer-based language model on a single Agatha Christie novel and generate stylistically consistent text from a user-supplied prompt.

## Abstract

Two classifiers were trained on CUB-200-2011 (200 species, 11,788 images). An **EfficientNet-B3** backbone, pretrained on ImageNet and fine-tuned with discriminative learning rates, reached **77.8% test accuracy**. A **custom CNN trained from scratch** on the same data reached **12.7%**, illustrating concretely how much transfer learning contributes when training data per class is limited (~30 images/class). For text generation, a 4-layer Transformer encoder with causal masking was trained at word level on *The Secret Adversary* and produces prompt-conditioned text in a recognisably similar register.

## What was achieved

| Task | Model | Approach | Test Accuracy | Precision | Recall | F1 (macro) |
|------|-------|----------|--------------:|----------:|-------:|-----------:|
| 1 | EfficientNet-B3 | Transfer learning (ImageNet → fine-tune) | **77.8%** | 78.8% | 78.0% | **0.779** |
| 1 | Custom CNN | Designed & trained from scratch | 12.7% | 13.0% | 12.8% | 0.107 |
| 2 | Transformer LM | Word-level, trained from scratch | — (generative) | — | — | — |

**Honest read of these numbers.** The EfficientNet result is a solid transfer-learning baseline. The custom CNN is deliberately reported as-is: a from-scratch network on a 200-class fine-grained task with only ~5,400 training images cannot compete with a pretrained backbone, and the 65-point gap is the most instructive finding in the project. Both results are reproducible from the committed notebook and weights (seed = 42).

### Data splits

The provided training set was split 9:1 into train/validation; the provided test set was held out entirely and never used for training.

| Split | Images |
|-------|-------:|
| Train | 5,394 |
| Validation | 600 |
| Test | 5,794 |

## Repository structure

```
DEEP-LEARNING/
├── README.md                  ← you are here (project overview)
├── CLEAN/                     ← final, presentable submission
│   ├── README.md              ← how to run / reproduce
│   ├── DeepLearning.ipynb     ← end-to-end notebook (Tasks 1 & 2)
│   ├── best_efficientnet_b3_cub.pth   ← EfficientNet-B3 weights (77.8%)
│   ├── best_custom_model.pth          ← custom CNN checkpoint
│   ├── model.pth                      ← Transformer LM weights
│   ├── DL_TASK_1.pdf          ← short report, Task 1
│   └── DL_TASK_2.pdf          ← short report, Task 2
└── RAW/                       ← dataset + working artifacts
    ├── README.md              ← dataset & coursework details
    ├── DeepLearning.ipynb     ← working copy of the notebook
    ├── 61262-0.txt            ← Agatha Christie text corpus (Task 2)
    ├── Coursework+2026.pdf    ← coursework brief
    └── CUB_200_2011/          ← Caltech-UCSD Birds dataset
```

## Tech stack

PyTorch · timm (EfficientNet-B3) · torchvision · scikit-learn · pandas · NumPy · Matplotlib / seaborn. Developed in Google Colab (CUDA GPU).

## Quick links

- **To run or reproduce the models** → see [`CLEAN/README.md`](CLEAN/README.md)
- **For dataset provenance and the coursework brief** → see [`RAW/README.md`](RAW/README.md)

## Acknowledgements

Dataset: Wah, Branson, Welinder, Perona & Belongie, *The Caltech-UCSD Birds-200-2011 Dataset* (2011). Text corpus: Agatha Christie via Project Gutenberg. Coursework: CSC8637, Newcastle University.
