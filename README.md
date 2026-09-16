# MRI Recognition with VAE and UNet

This project implements deep learning methods for brain MRI representation learning and segmentation using the preprocessed OASIS dataset.

The project was developed as part of COMP3710 and contains two main components:

- **Variational Autoencoder (VAE)** — learns a compact latent representation of brain MRI slices and reconstructs images from the learned latent space.
- **UNet** — performs multi-class semantic segmentation of MRI images using categorical pixel-wise predictions.

## Dataset

The project uses the preprocessed OASIS brain MRI dataset provided for COMP3710.

| Split | Images |
|---|---:|
| Training | 9,664 |
| Validation | 1,120 |
| Test | 544 |

MRI slices are grayscale images with a resolution of `256 × 256`.  
The segmentation masks contain four categorical classes.

## Models

### Variational Autoencoder

The VAE uses a convolutional encoder and decoder to learn a latent representation of MRI images.

The training objective combines reconstruction loss with KL divergence. The learned latent space is visualised to examine the structure captured by the encoder.

### UNet

The UNet follows an encoder-decoder architecture with skip connections for multi-class MRI segmentation.

The model produces four output channels and converts the predictions to categorical one-hot segmentation masks.

Training uses a combination of **Cross-Entropy Loss** and **Dice Loss**, while segmentation performance is evaluated using the Dice Similarity Coefficient for each class.

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

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- UQ Rangpur GPU Cluster
