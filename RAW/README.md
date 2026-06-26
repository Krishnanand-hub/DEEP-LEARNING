# RAW — Dataset, Corpus & Coursework Brief

This directory holds the inputs the project runs on: the bird image dataset, the text corpus for the language model, the original coursework brief, and a working copy of the notebook. Nothing here is "cleaned up" — it's the raw working material.

## Contents

```
RAW/
├── README.md                 ← this file
├── DeepLearning.ipynb        ← working copy of the notebook (same pipeline as CLEAN)
├── 61262-0.txt               ← text corpus for Task 2
├── Coursework+2026.pdf       ← the CSC8637 coursework brief
└── CUB_200_2011/             ← the image dataset (Task 1)
    ├── images/               ← 11,788 images in 200 species subfolders
    ├── images.txt            ← image_id → filepath
    ├── image_class_labels.txt← image_id → class label (1–200)
    ├── train_test_split.txt  ← image_id → is_train (1/0)
    ├── classes.txt           ← class_id → species name
    ├── bounding_boxes.txt    ← per-image bounding boxes
    ├── parts/                ← 15 part locations per image
    ├── attributes/           ← 322 binary attribute labels
    └── README                ← original dataset documentation
```

## Dataset 1 — Caltech-UCSD Birds-200-2011 (Task 1)

**What it is.** A fine-grained image classification benchmark: **200 bird species, 11,788 images**, roughly 30 images per class. Each image carries a class label, a bounding box, 15 annotated part locations, and 322 binary attribute labels (only the class labels and the train/test split are used in this project).

**Source.** The official dataset is published by Caltech Vision:

- Records page: `https://data.caltech.edu/records/65de6-vp158`
- Project site: `http://www.vision.caltech.edu/visipedia`

> **Provenance note.** The coursework brief instructs downloading the dataset from the Caltech records page (mirrored on Newcastle Canvas). CUB-200-2011 is also widely mirrored on **Kaggle** under the name *CUB_200_2011* — if you obtained it there, it is the same dataset; just confirm it's the **2011** release (200 classes, 11,788 images), not the older 2010 version (which has fewer images and a different split).

**How the project uses it.** The notebook merges three files — `images.txt`, `image_class_labels.txt`, and `train_test_split.txt` — into one DataFrame, shifts labels to 0-indexed, then:

- uses the provided **test set** (5,794 images) untouched for final evaluation, and
- splits the provided **training set** 9:1 into train (5,394) and validation (600).

Per the coursework rules, **the test set is never used for training.**

**Citation.**
> C. Wah, S. Branson, P. Welinder, P. Perona, S. Belongie. *The Caltech-UCSD Birds-200-2011 Dataset.* Computation & Neural Systems Technical Report, CNS-TR-2011-001. California Institute of Technology, 2011.

## Dataset 2 — Agatha Christie text corpus (Task 2)

**`61262-0.txt`** is a plain-text novel by Agatha Christie (*The Secret Adversary*), obtained from **Project Gutenberg** (file ID 61262). The Transformer language model is trained at word level on this single book, so generated text follows her early-20th-century mystery style. The notebook collapses whitespace, lowercases the text, builds a vocabulary from unique words, and encodes the corpus into integer sequences of length 40.

> Project Gutenberg texts are public domain in the US; check the licence header inside the file before redistributing.

## Coursework brief — `Coursework+2026.pdf`

The assignment is **CSC8637 "Deep Learning"**, Newcastle University, hand-in 20 February 2026. Relevant instructions:

**Task 1 — Fine-Grained Classification (50%).**
- Train a network to classify the 200 bird species.
- Produce **two** models: one based on a standard architecture (ResNet/VGG/DenseNet/Inception/EfficientNet, transfer learning encouraged) and **one of your own design**.
- Train only on the training set; a 9:1 train/val split is the suggested ratio. The test set is for final accuracy only and **must not** be used in training.
- Deliverables: a half-page report on data processing, hyper-parameters and the reasons behind them; full reproducible training/test code with run instructions (seed your RNGs); and model checkpoints.

**Task 2 — Text Generation.**
- Build a model using recurrent or transformer layers that generates text from a prompt at inference time.
- Trained on Agatha Christie's writing, so output should echo her style.
- Deliverables: a half-page report (with example generations), full runnable code that takes user words as input and outputs a sentence or more, and model checkpoints. Marking also asks for a comparison against an LLM (e.g. ChatGPT) producing Christie-style text.

