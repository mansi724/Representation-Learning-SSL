# Representation Learning — SimCLR vs MAE

Comparative analysis of self-supervised representation
learning paradigms on the Obstacle Detection Dataset.

## Overview
This project implements and compares two fundamental
self-supervised learning frameworks:

- **SimCLR** (Contrastive Learning) - learns by comparing
  two augmented views of the same image, pulling similar
  images together and pushing different images apart
  in embedding space.

- **MAE** (Masked Autoencoder) - learns by masking 75%
  of image patches and reconstructing the missing pixels
  using a Vision Transformer encoder-decoder architecture.

Both models are trained without any label supervision on
the Obstacle Detection Dataset containing 25 object classes.

---

## Dataset
[Obstacle Detection Dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset)

| Split | Images | Classes |
|-------|--------|---------|
| Train | 19,186 | 25 |
| Validation | 3,511 | 25 |
| Test | 1,629 | 25 |

Labels were used only for evaluation - not during training.

---

## Model Architectures

### SimCLR
- Backbone: ResNet18 trained from scratch
- Projection Head: Linear(512-512-128) + ReLU
- Loss: NT-Xent (Temperature=0.5)
- Optimizer: Adam (lr=3e-4)
- Epochs: 30

### MAE
- Encoder: ViT (depth=6, heads=3, dim=192)
- Decoder: Lightweight transformer (depth=2, dim=96)
- Mask ratio: 75% (147 out of 196 patches hidden)
- Patch size: 16x16
- Loss: MSE on masked patches only
- Optimizer: AdamW (lr=1.5e-4, weight_decay=0.05)
- Epochs: 30

---

## Evaluation Results

| Metric | SimCLR | MAE | Winner |
|--------|--------|-----|--------|
| KNN Accuracy @1 | 7.03% | 6.55% | SimCLR |
| KNN Accuracy @20 | 11.25% | 11.71% | MAE |
| Silhouette Score | 0.491 | 0.138 | SimCLR |
| Davies-Bouldin | 1.364 | 2.092 | SimCLR |
| Alignment | 1.969 | 0.539 | MAE |
| Uniformity | -3.278 | -0.872 | SimCLR |
| Retrieval @20 | 7.71% | 7.76% | MAE |
| Linear Probing | 13.76% | 14.18% | MAE |
| Intra/Inter Ratio | 1.150 | 0.997 | SimCLR |
| **Final Score** | **5** | **4** | **SimCLR** |

Both models are 3-3.5x above random baseline of 4%
without any label supervision during training.

---

## Key Findings
- SimCLR wins overall with a score of 5-4
- SimCLR forms better clusters - visible in UMAP plot
  with distinct separated groups
- MAE groups similar images closer together - better
  alignment and linear probing
- Both models learn complementary representations -
  neither dominates completely
- MAE improves more with additional training - wins
  KNN at larger K values

---



## Project Structure

    Representation_Learning.ipynb
    results/
        tsne_comparison.png
        umap_comparison.png
        mae_reconstruction.png
        metrics_comparison.png
        SimCLR_retrieval_grid.png
        MAE_retrieval_grid.png
    README.md

---

## How to Run
1. Open Representation_Learning.ipynb in Google Colab
2. Mount Google Drive
3. Download dataset from Kaggle
4. Run all cells in order


---

## Dependencies
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib
- umap-learn
- Pillow

---


## References
1. Chen et al. (2020) - SimCLR - https://arxiv.org/abs/2002.05709
2. He et al. (2021) - MAE - https://arxiv.org/abs/2111.06377
3. Wang and Isola (2020) - Alignment and Uniformity - https://arxiv.org/abs/2005.10242
4. Van der Maaten and Hinton (2008) - t-SNE
5. McInnes et al. (2018) - UMAP - https://arxiv.org/abs/1802.03426
6. Obstacle Detection Dataset - https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset
