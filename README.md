<div align="center">

# 🔬 U-Net Semantic Segmentation

### *Capacity and Ablation Analysis for Image Segmentation*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue?style=for-the-badge)](LICENSE)

*Critical evaluation of U-Net architecture for semantic image segmentation through controlled experiments analyzing channel capacity and skip connections.*

</div>

## 📖 Task Introduction

Semantic segmentation is the process of classifying each pixel of an image into a specific category, allowing for the detailed spatial understanding necessary to accurately delimit objects. In this project, the goal is to segment coins in synthetic images using the U-Net architecture, known for its effectiveness due to its symmetric contraction and expansion structure.

## 🏗️ Detailed Model Architecture

The model is based on three fundamental components defined in the implementation:

### 1. Encoder (Contracting Path)

* **Objective:** Capture context and extract a hierarchy of high-level features while reducing spatial dimensions.
* **Mechanism:** Utilizes repeated blocks of two 3x3 convolutions, each followed by Batch Normalization and ReLU activation.
* **Downsampling:** Employs 2x2 Max Pooling layers to halve height and width after every block, while simultaneously doubling the number of filters.

### 2. Decoder (Extending Path)

* **Objective:** Reconstruct spatial resolution and map the abstract features back to the original image dimensions for pixel-wise classification.
* **Mechanism:** Begins with 2x2 Transposed Convolutions (upsampling), followed by the same double 3x3 convolutional blocks used in the encoder.
* **Classification Layer:** A final 1x1 convolution reduces the feature map to a single channel, followed by a sigmoid-like activation to produce the probability mask.

### 3. Skip Connections (Concatenation)

* **Objective:** Recover spatial "lost" information during the downsampling process.
* **Mechanism:** Directly concatenate the high-resolution feature maps from the contracting path with the upsampled feature maps in the extending path.
* **Technical Impact:** This bridge allows the network to combine precise spatial details from the encoder with semantic, contextual information from the decoder, which is crucial for sharp edge definition.

## 🚀 Phase 1: Complexity & Simplification Study (Channels)

A progressive reduction in the number of filters at each stage was performed to observe the "Pareto frontier" between model efficiency and predictive performance.

Aquí tienes la tabla corregida y optimizada para que se visualice perfectamente en cualquier visor de Markdown (como GitHub). He ajustado las columnas para que coincidan con los encabezados y he refinado el análisis técnico basado en los resultados de tus experimentos.

### 🚀 Phase 1: Complexity & Simplification Study (Channels)

| Configuration | Channels (Base) | Result | Technical Analysis |
| --- | --- | --- | --- |
| **Baseline** | 64 | **Excellent** | Absolute edge precision; no artifacts. |
| **Model 1/2** | 32 | **Excellent** | Performance parity; confirms over-parameterization in the baseline. |
| **Model 1/4** | 16 | **Very Good** | Minor smoothing on fine textures, but mask integrity remains high. |
| **Model 1/8** | 8 | **Degradation** | Emergence of "noise" in the background; struggle with variance. |
| **Model 1/16** | 4 | **Poor** | Blurry masks and significant loss of circular geometry. |
| **Model 1/32** | 2 | **Underfitting** | Complete failure to converge; capacity is insufficient. |

**Key Finding:** The U-Net architecture exhibits massive redundancy for simple geometric datasets. A model with only **6.25% (1/16) of the original parameters** can still produce a recognizable segmentation mask.

## 🔍 Phase 2: Structural Ablation Study (Skip Connections)

This phase systematically disabled the "bridges" between the encoder and decoder to quantify their impact on spatial reconstruction.

### 1. Hierarchical Degradation Analysis

* **Deep Level (Level 4) Removal:** Disabling the deepest skip connection (closest to the bottleneck) had negligible impact. At this depth, features are highly abstract, and the bottleneck (1024 channels) carries sufficient semantic information to recover these features during upsampling.
* **Intermediate Level (Levels 3 & 2) Removal:** Performance remained surprisingly stable. However, a slight increase in false positives (background noise) was observed, indicating that intermediate connections help in "filtering" irrelevant background features.
* **Shallow Level (Level 1) Influence:** This connection (linking the 512x512 encoder output to the final decoder stage) proved to be the "anchor" for geometric precision. Even with other connections disabled, this level provides the high-resolution "template" required for sharp edges.

### 2. Pure Encoder-Decoder (No Skip 4)

* **Execution:** All connections disabled, forcing all information through the bottleneck.
* **Result:** While the model still identified the coins, the masks became "globular" and lost the crisp definition of the baseline.
* **Implication:** The bottleneck is powerful enough to classify the "what," but the skip connections are essential to define the "where" with sub-pixel accuracy.

## 📈 Project Conclusions

1. **Architecture vs. Brute Force:** structural design (skip connections) is more critical for segmentation quality than raw parameter count (channels). A "thin" U-Net with skips often outperforms a "thick" Encoder-Decoder without them.
2. **Resource Optimization:** For production environments involving high-contrast, low-complexity imagery, developers should prioritize smaller channel counts (Model 1/4) to save memory and inference time.
3. **The Power of the Bridge:** Skip connections effectively solve the "vanishing detail" problem in deep networks, allowing the decoder to "look back" at the original image's high-frequency details.

## 🛠️ Technologies and Tools

* **Framework:** PyTorch (v2.x) - handling the computational graph and backpropagation.
* **Optimization:** Adam Optimizer ($lr=1e-3, weight\_decay=1e-5$) for robust convergence.
* **Loss Function:** MSE Loss used for binary segmentation tasks on synthetic data.
* **Hardware:** NVIDIA CUDA acceleration for high-epoch training.
* **Visualization:** Matplotlib for visual comparison of ground truth and predictions.

## 👥 Authors

This project was created by [ArtHead](https://github.com/ArtHead-Devs), featuring:

* **👨‍💻 Fabio Nesta Arteaga Cabrera**: [NestX10](https://github.com/NestX10)
* **👨‍💻 Pablo Cabeza Lantigua**: [pabcablan](https://github.com/pabcablan)

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for more details.
