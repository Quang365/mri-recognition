# MRI Recognition with VAE and U-Net

This project implements deep learning methods for brain MRI representation learning and semantic segmentation using the preprocessed OASIS dataset.

The project was developed for COMP3710 and contains two main components:

- **Variational Autoencoder (VAE)** — learns a compact probabilistic latent representation of brain MRI slices and reconstructs images from the learned latent space.
- **U-Net** — performs four-class semantic segmentation of MRI images using pixel-wise categorical predictions.

## Dataset

The project uses the preprocessed OASIS brain MRI dataset provided for COMP3710.

| Split | Images |
|---|---:|
| Training | 9,664 |
| Validation | 1,120 |
| Test | 544 |

MRI slices are grayscale images with a resolution of `256 × 256` pixels and are normalized to the range `[0, 1]`.

For the segmentation task, the original mask values are mapped to four categorical classes:

```text
0   → Class 0
85  → Class 1
170 → Class 2
255 → Class 3
```

---

## 1. Variational Autoencoder (VAE)

### Architecture

The VAE uses a convolutional encoder-decoder architecture with a latent dimension of `16`.

The encoder progressively downsamples each MRI image:

```text
Input: 1 × 256 × 256

1   → 32
32  → 64
64  → 128
128 → 256
256 → 256

Final encoder feature map: 256 × 8 × 8
```

The encoded features are flattened and mapped to two 16-dimensional vectors:

- `mu` — mean of the latent distribution
- `logvar` — logarithm of the latent variance

A latent vector is sampled using the reparameterization trick:

```text
z = mu + std * epsilon
```

where `epsilon` is sampled from a standard normal distribution.

The decoder then uses transposed convolutions to reconstruct the original `256 × 256` MRI image.

### Training Objective

The VAE is trained using a combination of reconstruction loss and KL divergence:

```text
Total Loss = Reconstruction Loss + KL Divergence
```

The reconstruction term uses Mean Squared Error (MSE) to encourage the reconstructed MRI to match the original image.

The KL divergence regularizes the learned latent distributions toward a standard normal distribution.

### Training Configuration

| Parameter | Value |
|---|---:|
| Latent dimension | 16 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Training epochs | 30 |

The best model checkpoint was selected according to validation loss.

The lowest validation loss was obtained at **epoch 11**:

```text
Best Validation Loss: 302.5155
Best Epoch: 11
```

### Latent Space Analysis

After training, the mean latent representation (`mu`) was extracted for all test images.

```text
Test samples:       544
Latent dimensions:   16
Latent matrix:      544 × 16
```

Since the learned latent space is 16-dimensional, PCA was applied after training to project the latent representations into two dimensions for visualization.

The first two principal components explained:

| Component | Explained Variance |
|---|---:|
| PC1 | 21.80% |
| PC2 | 16.00% |
| **Total** | **37.80%** |

The 2-D PCA plot therefore provides a partial visualization of the structure learned in the full 16-dimensional VAE latent space.

---

## 2. U-Net Semantic Segmentation

### Architecture

The U-Net performs four-class semantic segmentation of the MRI images.

The network follows an encoder-decoder structure:

```text
Input: 1 × 256 × 256

Encoder:
32 → 64 → 128 → 256

Bottleneck:
512 channels

Decoder:
256 → 128 → 64 → 32

Output:
4 × 256 × 256
```

The encoder progressively reduces spatial resolution while learning higher-level features.

The decoder restores the original spatial resolution using transposed convolutions. Skip connections concatenate encoder features with the corresponding decoder features to preserve spatial information that may be lost during downsampling.

A final `1 × 1` convolution converts the decoder features into four class logits for every pixel.

### Categorical Output

For each input MRI, the network produces:

```text
[B, 4, 256, 256]
```

where the four channels represent the four segmentation classes.

The predicted class at each pixel is obtained using `argmax`, and predictions can also be represented as categorical one-hot masks:

```text
[B, 4, 256, 256]
```

with exactly one active class per pixel.

### Training Objective

Training uses a combination of Cross-Entropy Loss and Soft Dice Loss:

```text
Total Loss = Cross-Entropy Loss + Soft Dice Loss
```

Cross-entropy provides pixel-wise classification supervision, while Dice loss directly encourages overlap between predicted and ground-truth segmentation regions.

Dice Similarity Coefficient is defined as:

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       --------------------------------
       |Prediction| + |Ground Truth|
```

A Dice score of `1.0` represents perfect overlap.

### Class Distribution

The segmentation masks are imbalanced, with Class 0 representing most pixels:

| Class | Approx. Pixel Frequency |
|---|---:|
| Class 0 | 72.26% |
| Class 1 | 5.59% |
| Class 2 | 11.42% |
| Class 3 | 10.73% |

For this reason, segmentation performance is evaluated using Dice scores separately for each class rather than relying only on overall pixel accuracy.

### Training Configuration

| Parameter | Value |
|---|---:|
| Input size | 256 × 256 |
| Number of classes | 4 |
| Batch size | 8 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross-Entropy + Soft Dice |

Model selection is based on the **minimum validation Dice across all four classes**, ensuring that strong performance on one class does not hide poor segmentation of another class.

### Validation Results

The selected checkpoint achieved:

| Class | Validation Dice |
|---|---:|
| Class 0 | 0.9994 |
| Class 1 | 0.9596 |
| Class 2 | 0.9650 |
| Class 3 | 0.9787 |
| **Minimum Dice** | **0.9596** |

All four segmentation classes achieved a validation Dice score above `0.90`.

### Test Results

Final evaluation on the 544-image test set produced:

| Class | Test Dice |
|---|---:|
| Class 0 | 0.9993 |
| Class 1 | 0.9630 |
| Class 2 | 0.9645 |
| Class 3 | 0.9788 |
| **Minimum Dice** | **0.9630** |
| **Mean Dice** | **0.9764** |

The final test loss was:

```text
Test Loss: 0.0554
```

All four classes achieved Dice scores above `0.96` on the test set.

---

## Project Structure

```text
mri-recognition/
├── notebooks/
│   ├── VAE.ipynb
│   └── UNet.ipynb
├── checkpoints/
├── results/
├── .gitignore
└── README.md
```

Model checkpoints and the OASIS dataset are excluded from the repository due to their size.

---

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- UQ Rangpur GPU Cluster
