# k-sparse-autoencoder
# Top-K Autoencoder for Fashion MNIST

An autoencoder implementation using TensorFlow/Keras with a custom Top-K layer to enforce sparsity in the encoding layer. Designed for feature learning on the Fashion MNIST dataset.

## Overview
This project implements an autoencoder with a custom **Top-K layer** that retains only the top `k` activations during forward propagation. The model is trained on the Fashion MNIST dataset to reconstruct images while learning sparse representations in the bottleneck layer.

## Features
- **Custom Top-K Layer**: Dynamically masks all but the top `k` activations to encourage sparsity.
- **Autoencoder Architecture**:
  - Encoder: Reduces input from 784 → 128 (ReLU) → 64 (Linear).
  - Top-K Layer: Retains top 32/64 activations.
  - Decoder: Reconstructs from 32 → 128 (ReLU) → 784 (Sigmoid).
- **Training**: 300 epochs with Adam optimizer and MSE loss.
- **Visualization**: Training/validation loss curves to monitor model performance.

## Installation
Dependencies:
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Fashion MNIST dataset (loaded via `keras.datasets`)

Install requirements:
```bash
pip install tensorflow numpy matplotlib
