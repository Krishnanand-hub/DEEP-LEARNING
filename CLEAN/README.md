# CLEAN — Final Submission & How to Run

This directory holds the presentable version of the project: one end-to-end notebook, the trained model checkpoints, and the two short reports. If you cloned this repo and want to reproduce the results, **start here**.

## What's in this directory

```
CLEAN/
├── README.md                        ← this file
├── DeepLearning.ipynb               ← full pipeline: Tasks 1 & 2
├── best_efficientnet_b3_cub.pth     ← EfficientNet-B3 weights  → 77.8% test acc
├── best_custom_model.pth            ← custom CNN checkpoint (dict: model + optimizer + epoch)
├── model.pth                        ← Transformer LM weights (Task 2)
├── DL_TASK_1.pdf                    ← half-page report, image classification
└── DL_TASK_2.pdf                    ← half-page report, text generation
```

## What the notebook does

| Section | Content |
|---------|---------|
| Task 1a | Loads CUB-200-2011, builds a `Dataset`/`DataLoader`, fine-tunes **EfficientNet-B3** (pretrained) with discriminative LRs, mixed precision, cosine schedule. Evaluates accuracy/precision/recall/F1 + confusion matrix. |
| Task 1b | Defines and trains a **custom CNN from scratch**, evaluates on the same test set. |
| Task 2 | Word-level tokenization of an Agatha Christie novel, trains a **4-layer Transformer encoder** with causal masking, then generates text from a user prompt. |

## Prerequisites

- Python 3.10+
- A CUDA-capable GPU is strongly recommended (the notebook auto-falls back to CPU, but training will be impractically slow).
- The notebook was written for **Google Colab**. Paths and the Hugging Face / timm download step assume that environment.

### Dependencies

There is no `requirements.txt` in the repo yet (add one — see "Known issues"). Install manually:

```bash
pip install torch torchvision timm scikit-learn pandas numpy matplotlib seaborn pillow
```

## How to run — step by step

> ⚠️ The notebook currently uses **hardcoded Google Colab paths** such as
> `/content/drive/MyDrive/DL/bird_CUB_200_2011.zip` and
> `/content/birds/CUB_200_2011`. You must edit these to match wherever you put the data.

1. **Get the dataset.** Obtain `CUB_200_2011` (see [`../RAW/README.md`](../RAW/README.md) for the source). The dataset is already committed under `../RAW/CUB_200_2011/`, or download a fresh copy.

2. **Fix the data paths.** In the notebook, set the data root to your actual dataset location, e.g.:
   ```python
   data_root = "../RAW/CUB_200_2011"   # or wherever you extracted it
   ```
   If you keep the original zip-extraction cell, point `zip_path` at your archive and `extract_path` at a writable folder.

3. **Set the corpus path (Task 2).** Point the text path at the Agatha Christie file:
   ```python
   file_path = "../RAW/61262-0.txt"
   ```

4. **Run Task 1 — EfficientNet.** Execute the cells through training. With the fixed seed (42) and the provided hyper-parameters you should land near **77.8% test accuracy**. Training the full 45 epochs on a single GPU takes time; plan accordingly.

5. **Run Task 1 — Custom CNN.** Execute the from-scratch CNN cells. Expect **~12–13% accuracy** — this is the documented result, not a bug.

6. **Run Task 2 — Text generation.** Train the Transformer (15 epochs), then run the final cell, which prompts:
   ```
   Enter prompt: <type a few words>
   ```
   and prints generated text.

### Reproducing without retraining

To skip training and just evaluate, load the committed checkpoints:

```python
# EfficientNet-B3 (Task 1)
model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=200)
model.load_state_dict(torch.load("best_efficientnet_b3_cub.pth",
                                 map_location=device, weights_only=True))

# Custom CNN (checkpoint is a dict)
ckpt = torch.load("best_custom_model.pth", map_location=device, weights_only=True)
custom_model.load_state_dict(ckpt["model_state_dict"])
```

> **Note on `weights_only=True`:** `torch.load` uses Python pickle and can execute arbitrary code from a malicious checkpoint. Always pass `weights_only=True` when loading weights you didn't produce.

## Known issues & gotchas

- **Hardcoded Colab paths** — the single most common reason a clone fails to run. Edit them first (steps 2–3).
- **`best_custom_model.pth` is a dictionary, not a bare state_dict.** Load `ckpt["model_state_dict"]`, not `ckpt` directly, or you'll get a key error.
- **EfficientNet expects 300×300 input** (matching the training transform). Feeding other sizes will degrade or break inference.
- **`pretrained=True` needs internet** the first time (timm downloads ImageNet weights). For offline reproduction use `pretrained=False` and load the committed `.pth`.
- **The custom CNN's low accuracy is expected** — do not "fix" it by accident; it's a deliberate from-scratch baseline.
- **No `requirements.txt` / no pinned versions.** Results are seed-fixed but not environment-pinned; minor numeric drift across library versions is possible.

## Results recap

| Model | Test Accuracy | F1 (macro) |
|-------|--------------:|-----------:|
| EfficientNet-B3 (transfer learning) | 77.8% | 0.779 |
| Custom CNN (from scratch) | 12.7% | 0.107 |
| Transformer LM | generative (qualitative) | — |
