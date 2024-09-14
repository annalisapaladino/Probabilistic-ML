import torch
import torchvision
import PIL
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm
import numpy as np

# FacesDataset class: custom dataset for handling face images
class FacesDataset(torch.utils.data.Dataset):
    # __init__ function: takes a list of image paths as input and sets up image transformations
    # Images are resized to 128×128 and converted into a PyTorch tensor using Resize and ToTensor
    def __init__(self, images): 
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        self.images = images

    # __len__ function: returns the length of the dataset (number of images)
    def __len__(self):
        return len(self.images)

    # __getitem__ function: loads an image at a specific index, applies the transformation, and returns the transformed image as both the input (x) and the target (y)
    def __getitem__(self, idx):
        image = PIL.Image.open(self.images[idx]).convert('RGB')
        x = self.transform(image)
        y = x
        return x, y
    


# Training function: trains the VAE model on the training and validation datasets (train_loader, val_loader)
def train_model(
    train_loader : DataLoader,
    val_loader : DataLoader,
    model : nn.Module, # VAE model
    n_epochs : int = 100,
    initial_lr : float = 1e-3,
    end_lr : float = 1e-5,
    clip_grad_value : float = 1.0, # Clip gradients to avoid exploding gradients
    device : torch.device = torch.device('cpu')
):
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=end_lr/initial_lr, total_iters=n_epochs) # linear learning rate scheduler (lr_scheduler.LinearLR) to adjust the learning rate from initial_lr to end_lr over n_epochs
    
    # Loss function: sum of MSE and KLD
    # MSE (Mean Squared Error): Measures the reconstruction error between the original image x and the reconstructed image x_recon
    # KLD (Kullback-Leibler Divergence): Ensures that the latent space follows a standard normal distribution
    def vae_loss(x, x_recon, mu, logvar):
        MSE = nn.functional.mse_loss(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
    criterion = vae_loss
    
    train_losses = []
    val_losses = []
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        model.train()
        train_loss = 0
        n_samples = 0
        for x,y in train_loader:
            # For each batch of images, performs a forward pass through the model, computes the loss, and performs backpropagation using loss.backward()
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            recon_x, mu, logvar = model(x)
            loss = criterion(recon_x, y, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value) # Clips gradients to avoid exploding gradients

            # Update weights
            optimizer.step()

            train_loss += loss.item()
            n_samples += len(x)

        train_loss /= n_samples
        train_losses.append(train_loss)

        # Compute test loss
        model.eval()
        val_loss = 0
        n_samples = 0
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                recon_x, mu, logvar = model(x)
                loss = criterion(recon_x, y, mu, logvar)

                val_loss += loss.item()
                n_samples += len(x)

        val_loss /= n_samples
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

    return train_losses, val_losses, model


# Helper Function for Hyperparameter Tuning: trains the VAE model for a given set of hyperparameters (model parameters and training parameters) and returns the average validation loss over the last few epochs
def ht_function(
        train_loader : DataLoader,
        val_loader : DataLoader,
        model_class : nn.Module,
        model_params : dict,
        training_params : dict,
        device : torch.device = torch.device('cpu')
):
    # The model's hidden_dim and latent_dim are rounded before being passed to the model
    model_params['hidden_dim'] = round(model_params['hidden_dim'])
    model_params['latent_dim'] = round(model_params['latent_dim'])

    model = model_class(**model_params).to(device)
    _, val_losses, _ = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        **training_params
    )
    
    last_n_epochs = 3
    last_train_loss = np.mean(val_losses[-last_n_epochs:])

    return last_train_loss

# Image Encoding Function: encodes an input image into its latent space representation (z) using the VAE's encoder
def encode_image(image, model, device):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(image.unsqueeze(0).to(device)) # The image is passed through the encoder to obtain the latent space distribution parameters mu and logvar
        z = model.reparameterize(mu, logvar).cpu().numpy() # The reparameterize function samples from this latent distribution to get the latent vector z
    return z

# Latent Space Interpolation: interpolates between two latent vectors z1 and z2 and generates a series of intermediate images
def interpolate(n_points, z1, z2, model, device):
    interpolated_images = []
    model.eval()
    with torch.no_grad():
        for alpha in np.linspace(0, 1, n_points):
            # For each interpolation step alpha, a new latent vector z is generated by linearly combining z1 and z2
            # The VAE's decoder then generates an image from this interpolated latent vector
            z = alpha*z1 + (1-alpha)*z2
            z = torch.tensor(z).to(device)
            image = model.decode(z).cpu().permute(0, 2, 3, 1).numpy()
            interpolated_images.insert(0, image[0])

    return interpolated_images # The generated images are returned as a list