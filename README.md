# Semantic Segmentation with U-Net Architecture: Capacity and Ablation Analysis

This project implements and critically evaluates the **U-Net** neural network architecture for semantic image segmentation. The work was developed as part of the **Machine Learning II** course at the **University of Las Palmas de Gran Canaria (ULPGC)**. Through a series of controlled experiments, we analyze the impact of the number of channels and the importance of skip connections on model accuracy.

## 📖 Task Introduction

Semantic segmentation is the process of classifying each pixel of an image into a specific category, allowing for the detailed spatial understanding necessary to accurately delimit objects. In this project, the goal is to segment coins in synthetic images using the U-Net architecture, known for its effectiveness due to its symmetric contraction and expansion structure.

## 🏗️ Model Architecture

The model is based on three fundamental components defined in the implementation:

1. **Encoder (Contracting Path):** Captures context and extracts high-level features through convolutional blocks and *Max Pooling* layers that progressively reduce resolution.
2. **Decoder (Extending Path):** Reconstructs spatial resolution to generate the segmentation mask using transposed convolutions.
3. **Skip Connections:** Directly link feature maps from the encoder to the decoder via concatenation. Their primary impact is preserving critical spatial information often lost during downsampling, drastically improving precision at the edges.

## 🚀 Phase 1: Simplification Experiments (Channels)

A progressive reduction in the number of filters at each stage was performed to observe the balance between model size and performance.

| Configuration | Channels | Result | Technical Analysis |
| --- | --- | --- | --- |
| **Baseline** | 64, 128, 256, 512 | **Excellent** | The model is capable of segmenting edges with absolute precision. |
| **Model 1/2** | 32, 64, 128, 256 | **Excellent** | Negligible loss of precision; the original model was over-parameterized. |
| **Model 1/4** | 16, 32, 64, 128 | **Very Good** | Performance is maintained, though very fine details show minor smoothing. |
| **Model 1/8** | 8, 16, 32, 64 | **Degradation** | Lack of filters makes it difficult to distinguish between background and object in complex areas. |
| **Model 1/16** | 4, 8, 16, 32 | **Poor** | Significantly low quality; blurry masks and loss of geometry. |
| **Model 1/32** | 2, 4, 8, 16 | **Underfitting** | Total collapse of learning capacity; the network fails to extract useful features. |

**Conclusion:** It is identified that the original network possesses excessive capacity for this specific dataset. The model can be reduced to **1/4 of its size** without compromising the integrity of the segmentation.

## 🔍 Phase 2: Ablation Study (Skip Connections)

A systematic removal of skip connections was performed to validate their importance within the architecture.

### 1. Progressive Removal

* **No Skip 1 (Deep Level):** By removing the connection closest to the *bottleneck*, the impact is minimal. The decoder recovers information through the standard upsampling process.
* **No Skip 2:** By removing the two deepest connections, the performance difference remains almost imperceptible. Critical spatial information seems to concentrate in the shallower layers.
* **No Skip 3:** With only the level 1 connection active, segmentation surprisingly shows no significant degradation, maintaining the spatial integrity of the masks.

### 2. Model No Skip 4 (Pure Encoder-Decoder)

In the final experiment, all skip connections were removed, transforming the network into a **standard Encoder-Decoder**.

* **Result:** The model continued to produce results very similar to the baseline.
* **Conclusion:** In this specific case, the **bottleneck** (1024 channels) is sufficient to capture and reconstruct the necessary features. The geometric simplicity and high contrast of the dataset allow the network to dispense with spatial "shortcuts" without a collapse in performance.

## 📈 Final Project Conclusions

1. **Structural Robustness:** U-Net proves to be extremely robust. In low-complexity visual datasets, the architecture can be massively simplified in both channel depth and connectivity.
2. **Parametric Efficiency:** It has been validated that a lighter model (1/4 channels) is preferable for optimizing computational resources without sacrificing accuracy metrics.
3. **Role of Levels:** While deep layers manage global semantics, shallow layers (especially Level 1) act as the primary anchor for geometric edge precision.

## 🛠️ Technologies and Tools

* **Framework:** PyTorch (torch.nn, torch.optim).
* **Processing:** NumPy, Torchvision.
* **Visualization:** Matplotlib for visual comparison of ground truth and predictions.

## 👥 Authors

This project was created by [ArtHead](https://github.com/ArtHead-Devs), featuring two members:

* **👨‍💻 Fabio Nesta Arteaga Cabrera**: [NestX10](https://github.com/NestX10)
* **👨‍💻 Pablo Cabeza Lantigua**: [pabcablan](https://github.com/pabcablan)

## 📄 License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for more details.