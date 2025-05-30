# 🧠 Variational Autoencoder for Facial Image Generation with Bayesian Optimization
Annalisa Paladino, Probabilistic Machine Learning - UniTS 2024
> An interpretable deep learning project for facial image generation and hyperparameter tuning using Bayesian methods.

## 📘 Project Overview

This project implements a **Variational Autoencoder (VAE)** to generate high-quality facial images from a dataset of Lego-style faces. The model is trained and optimized using **Bayesian Optimization**, which enables automatic tuning of hyperparameters to enhance performance and image fidelity.

### 🔍 Goals

- Compress and reconstruct facial images via VAE
- Learn a smooth and structured latent space
- Tune the VAE’s hyperparameters using Bayesian Optimization
- Enable applications such as facial interpolation and morphing

---

## 🧱 Architecture

### Variational Autoencoder (VAE)

The VAE architecture consists of:

- **Encoder**: Maps input images to a latent space distribution (mean and log-variance)
- **Reparametrization trick**: Allows backpropagation through the stochastic latent variables
- **Decoder**: Reconstructs the original image from a sampled latent vector

The objective is to minimize the combined **Reconstruction Loss (MSE)** and **Kullback-Leibler Divergence (KLD)**:


---

## 🔧 Hyperparameter Optimization

### Bayesian Optimization Framework

- **Search space**: Discrete and continuous hyperparameters (hidden layer size, latent dimension)
- **Objective**: Minimize validation loss over the VAE model
- **Surrogate model**: Gaussian Process (GP) with Matern kernel
- **Acquisition function**: Expected Improvement (EI)

#### Key Features

- Black-box optimization
- Global search for optimal settings
- Computationally efficient compared to grid/random search

---

## 🧪 Experimental Setup

- **Dataset**: Lego facial images
- **Optimizer**: Adam
- **Initial LR**: 1e-3 → **Final LR**: 5e-7 (with scheduling)
- **Epochs**: 400
- **Batch Size**: 64
- **Gradient Clipping**: 1.0

### Optimized Hyperparameters

| Parameter          | Range Explored | Optimal Value (Example) |
|--------------------|----------------|--------------------------|
| Hidden Dimension   | 128 – 2048     | 512                      |
| Latent Dimension   | 8 – 64         | 32                       |

---

## 🖼️ Results

- **Generated Images**: High-quality outputs that resemble training inputs
- **Latent Interpolation**: Smooth morphing between two facial images
- **Loss Curves**: Demonstrate learning stability and optimization efficacy

---

## 📈 Visualizations

- Evolution of hyperparameters during optimization
- Visual comparison of original vs reconstructed images
- Latent space traversals for image morphing

---

## 📚 Conclusion

This project shows that:
- VAE models are effective at encoding and generating structured facial data
- Bayesian Optimization automates hyperparameter tuning efficiently
- The model generalizes well and supports continuous transformations in the latent space

---

## 👩‍💻 Author

**Annalisa Paladino**  
_MSc in Probabilistic Machine Learning_

---

## 🛠️ Requirements

- Python 3.8+
- `torch`, `numpy`, `matplotlib`
- `scikit-learn`, `GPyOpt` or `BoTorch` (for Bayesian Optimization)

---

## 📂 Repository Structure

├── data/ # Dataset of Lego facial images
├── models/ # VAE architecture and training script
├── optimization/ # Bayesian optimization routines
├── notebooks/ # Jupyter notebooks for experiments
├── results/ # Generated samples and loss plots
└── README.md # Project documentation


---

## 🚀 Future Work

- Extend to other facial datasets (e.g., CelebA)
- Compare with other generative models (GANs, Diffusion Models)
- Incorporate disentangled representations



