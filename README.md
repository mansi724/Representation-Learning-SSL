# Representation Learning — SimCLR vs MAE

Comparative analysis of contrastive vs reconstruction-based 
self-supervised learning on the Obstacle Detection Dataset.

## Overview
This project implements and compares two self-supervised 
learning paradigms:
- **SimCLR** — contrastive representation learning
- **MAE** — masked autoencoder reconstruction learning

Both models are trained without any labels on the 
[Obstacle Detection Dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset) 
containing 25 object classes.

## Results

| Metric | SimCLR | MAE | Winner |
|--------|--------|-----|--------|
| KNN Accuracy @20 | 12.33% | 11.17% | SimCLR |
| Silhouette Score | 0.326 | 0.254 | SimCLR |
| Davies-Bouldin | 1.660 | 1.488 | MAE |
| Alignment | 1.931 | 0.779 | MAE |
| Uniformity | -3.096 | -0.938 | SimCLR |

**Final Score: SimCLR 3 — MAE 2**

## Key Findings
- SimCLR produces more discriminative embeddings — better KNN accuracy and uniformity
- MAE produces better aligned embeddings — similar images sit closer together
- Both models significantly outperform random baseline of 4% on 25 classes without any label supervision

## Project Structure

    Representation_Learning.ipynb
    results/
        tsne_comparison.png
        umap_comparison.png
        mae_reconstruction.png
        metrics_comparison.png
    README.md

## How to Run
1. Open Representation_Learning.ipynb in Google Colab
2. Mount Google Drive
3. Download dataset from Kaggle
4. Run all cells in order

## Models

**SimCLR**
- Backbone: ResNet18
- Embedding dim: 128
- Loss: NT-Xent (temperature=0.5)
- Epochs: 5

**MAE**
- Encoder: ViT (depth=6, heads=3, dim=192)
- Decoder: lightweight (depth=2, dim=96)
- Mask ratio: 75%
- Patch size: 16x16
- Epochs: 5

## Dependencies
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib
- umap-learn
- Pillow
