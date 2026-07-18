# Variational Autoencoder with Bayesian Hyperparameter Optimization

This repository contains a probabilistic machine learning project focused on image generation with a convolutional Variational Autoencoder (VAE). The model is trained on a dataset of LEGO-style face images and its main architectural hyperparameters are selected through Bayesian optimization.

The project was developed by **Annalisa Paladino** for the **Probabilistic Machine Learning** course at the University of Trieste in 2024.

## Project Overview

The project investigates three connected tasks:

1. learning a compact probabilistic representation of face images;
2. reconstructing and generating images through a convolutional VAE;
3. selecting the hidden and latent dimensions through Bayesian optimization.

After training, the learned latent space is also used for interpolation between two images, producing a continuous face-morphing sequence.

The complete experimental workflow is contained in [`VAE.ipynb`](VAE.ipynb), while the model architecture and reusable training utilities are separated into [`models.py`](models.py) and [`utils.py`](utils.py).

## Main Components

### Variational Autoencoder

The implemented model is a convolutional Variational Autoencoder. It consists of:

- a convolutional encoder;
- two fully connected branches that estimate the latent mean and log-variance;
- the reparameterization trick;
- a fully connected decoder;
- a sequence of transposed convolutions that reconstruct the image.

Given an input image `x`, the encoder estimates the mean and variance of a Gaussian latent distribution:

```text
q_phi(z | x) = Normal(mean = mu(x), covariance = diag(sigma(x)^2))
```

A latent vector is sampled with the reparameterization trick:

```text
epsilon ~ Normal(0, I)
z = mu + sigma * epsilon
```

The decoder then maps `z` back to the image space.

### Training Objective

The model is trained by minimizing the sum of two terms:

```text
total_loss = reconstruction_loss + KL_divergence
```

The reconstruction term is the squared distance between the input image and its reconstruction:

```text
reconstruction_loss = sum((x - x_reconstructed)^2)
```

The Kullback-Leibler divergence regularizes the approximate posterior toward a standard normal prior:

```text
KL_divergence = -0.5 * sum(1 + log_variance - mean^2 - exp(log_variance))
```

In the implementation, `log_variance` is stored directly, so `exp(log_variance)` corresponds to the latent variance.

This combination encourages the model to reconstruct the images accurately while learning a continuous and structured latent space.

## Model Architecture

Images are resized to `128 x 128` and represented as RGB tensors.

### Encoder

The encoder uses a sequence of convolutional blocks. Each block contains:

- a two-dimensional convolution;
- a `Tanh` activation;
- batch normalization.

The convolutional feature map is flattened and passed through a fully connected layer. Two separate networks then produce:

- the latent mean `mu`;
- the latent log-variance `logvar`.

### Decoder

The decoder first expands the sampled latent vector with fully connected layers. It then reshapes the representation and applies a sequence of transposed-convolution blocks.

Each intermediate decoder block contains:

- a transposed convolution;
- a `LeakyReLU` activation;
- batch normalization.

The final layer uses a sigmoid activation, producing pixel values in the interval `[0, 1]`.

### Tunable Architectural Parameters

The model exposes three principal parameters:

| Parameter | Description |
|---|---|
| `latent_dim` | Dimension of the probabilistic latent representation |
| `hidden_dim` | Width of the fully connected encoder and decoder representation |
| `dropout_rate` | Dropout probability applied before the encoder's hidden layer |

## Bayesian Hyperparameter Optimization

Training a VAE requires architectural choices that can have a substantial effect on reconstruction quality and latent-space structure. This project uses Bayesian optimization to search for suitable values of:

- `hidden_dim`;
- `latent_dim`.

The optimization uses the `bayesian-optimization` Python package. For each proposed configuration, a VAE is trained for 50 epochs and evaluated on the validation set. The objective returned to the optimizer is the negative average validation loss over the final three epochs, because the optimizer maximizes its objective.

The search space used in the notebook is:

| Hyperparameter | Search interval |
|---|---:|
| Hidden dimension | 128 to 2048 |
| Latent dimension | 8 to 64 |

The optimization procedure performs:

- 10 initial random evaluations;
- 90 Bayesian optimization iterations.

According to the stored notebook output, the best configuration found was approximately:

| Parameter | Selected value |
|---|---:|
| Hidden dimension | 426 |
| Latent dimension | 18 |
| Optimization target | -1000.3 |

Because model training is stochastic and depends on hardware, software versions and dataset handling, rerunning the notebook may produce different values.

## Final Training Configuration

After hyperparameter optimization, the selected model is retrained with the following settings:

| Setting | Value |
|---|---:|
| Image size | `128 x 128` |
| Batch size | 32 |
| Epochs | 400 |
| Initial learning rate | `1e-3` |
| Final learning rate | `5e-7` |
| Optimizer | Adam |
| Gradient clipping norm | 1.0 |
| Dropout rate | 0.05 |
| Random seed | 42 |

A linear learning-rate scheduler progressively reduces the learning rate over the complete training run.

The final model shown in the notebook contains approximately **10.44 million trainable parameters**. The stored run reports:

| Metric | Final value |
|---|---:|
| Training loss | 429.3 |
| Validation loss | 1108.3 |

These values are losses per sample, calculated from a summed pixel-wise MSE and KL-divergence term. They should therefore not be interpreted as normalized per-pixel errors.

## Dataset

The repository includes [`dataset.zip`](dataset.zip), containing 1,111 LEGO-style face images in JPEG format.

The archive currently stores the images inside the directory:

```text
easy_dataset/
```

The notebook, however, expects the images to be available inside:

```text
dataset/
```

Before running the notebook, extract the archive and rename or move the extracted folder so that the final structure is:

```text
Probabilistic-ML-main/
├── dataset/
│   ├── 3626ap01.jpg
│   ├── 3626apb03.jpg
│   └── ...
├── VAE.ipynb
├── models.py
└── utils.py
```

The custom `FacesDataset` class:

1. opens each image as RGB;
2. resizes it to `128 x 128`;
3. converts it to a PyTorch tensor;
4. returns the same image as both input and reconstruction target.

The notebook divides the dataset into training, validation and test subsets. The effective proportions are:

- 64% training;
- 16% validation;
- 20% test.

## Generated Outputs

The notebook demonstrates three main outputs.

### Random Sampling

Random latent vectors are sampled from a standard normal distribution:

```text
z ~ Normal(0, I)
```

The decoder transforms these vectors into synthetic LEGO-style face images. This experiment checks whether the regularized latent space can generate plausible samples without starting from a real input image.

### Image Reconstruction

Although the notebook primarily visualizes generated samples, the VAE training objective explicitly learns to reconstruct every input image. Reconstruction quality is monitored through the training and validation losses.

### Latent-Space Interpolation

Two dataset images are encoded into latent vectors `z1` and `z2`. Intermediate vectors are produced through linear interpolation:

```text
z(alpha) = alpha * z1 + (1 - alpha) * z2
where 0 <= alpha <= 1
```

Decoding these intermediate points produces a gradual transition between the two faces. The notebook creates both:

- a static sequence of interpolated images;
- an animated file named `interpolated_images.gif`.

The interpolation experiment provides a qualitative indication that the model has learned a reasonably continuous latent representation.

## Repository Structure

```text
Probabilistic-ML-main/
├── README.md
├── VAE.ipynb
├── models.py
├── utils.py
├── dataset.zip
└── slides.pdf
```

### File Descriptions

| File | Purpose |
|---|---|
| `VAE.ipynb` | Complete experimental workflow: loading, tuning, training, sampling and interpolation |
| `models.py` | Convolutional VAE architecture |
| `utils.py` | Dataset class, training loop, tuning objective, encoding and interpolation utilities |
| `dataset.zip` | Compressed LEGO-style face dataset |
| `slides.pdf` | Presentation associated with the project |
| `README.md` | Project documentation |

## Installation

Python 3.9 or later is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm bayesian-optimization tabulate jupyter
```

A CUDA-compatible GPU is strongly recommended for the full Bayesian search and the 400-epoch final training run. The code also supports CPU execution, but the complete workflow will be considerably slower.

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd Probabilistic-ML-main
```

Extract the dataset:

```bash
unzip dataset.zip
mv easy_dataset dataset
```

On systems where `unzip` is unavailable, extract the archive manually and rename the resulting `easy_dataset` folder to `dataset`.

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
VAE.ipynb
```

Run the notebook cells in order.

## Reusing the Model

A model can be instantiated directly from `models.py`:

```python
import torch
from models import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(
    latent_dim=18,
    hidden_dim=426,
    dropout_rate=0.05,
).to(device)
```

To train it, create the required data loaders and call:

```python
from utils import train_model

train_losses, val_losses, trained_model = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    n_epochs=400,
    initial_lr=1e-3,
    end_lr=5e-7,
    clip_grad_value=1.0,
    device=device,
)
```

To encode an image and interpolate between two latent representations:

```python
from utils import encode_image, interpolate

z1 = encode_image(image1, trained_model, device)
z2 = encode_image(image2, trained_model, device)

frames = interpolate(
    n_points=20,
    z1=z1,
    z2=z2,
    model=trained_model,
    device=device,
)
```

## Reproducibility

The notebook sets the NumPy and PyTorch random seeds to 42. This improves reproducibility, but it does not guarantee identical results across all environments.

Exact reproducibility may still be affected by:

- CUDA and cuDNN implementations;
- GPU model;
- PyTorch version;
- parallel data-processing behavior;
- stochastic latent sampling;
- Bayesian optimization package version.

The repository does not currently pin package versions. For fully reproducible experiments, a versioned `requirements.txt` or environment file should be added.

## Current Limitations

The repository is suitable as an academic experiment, but several aspects should be considered before treating it as a production-ready package.

### Computational Cost

The Bayesian search trains 100 separate models for 50 epochs each. This is computationally expensive, especially without a GPU.

### No Saved Checkpoint

The trained model is not stored in the repository. Running the sampling and interpolation sections requires retraining the model or adding code to load a saved checkpoint.

### Limited Evaluation

The project reports training and validation loss and provides qualitative generated samples. It does not include stronger generative-model metrics such as:

- Fréchet Inception Distance;
- Kernel Inception Distance;
- reconstruction error on the held-out test set;
- latent-space disentanglement measures.

### Dataset Path Requires Manual Adjustment

The directory name inside `dataset.zip` does not match the path expected by the notebook. The extracted directory must be renamed before execution.

### Hyperparameter Search Scope

Only the hidden dimension and latent dimension are optimized. Other influential parameters remain fixed, including:

- learning rate;
- dropout rate;
- batch size;
- KL-divergence weighting;
- convolutional architecture.

## Possible Extensions

Potential improvements include:

- saving and loading model checkpoints;
- adding a pinned `requirements.txt`;
- reporting test-set performance;
- visualizing original and reconstructed images side by side;
- introducing a weighted or annealed KL term;
- comparing the VAE with a beta-VAE;
- extending Bayesian optimization to training parameters;
- evaluating the latent space with quantitative metrics;
- comparing linear interpolation with spherical interpolation;
- testing the architecture on larger face datasets;
- reorganizing the code as an installable Python package.
