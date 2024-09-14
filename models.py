import torch
import torch.nn as nn

# conv_block class: defines a basic block for the encoder, which performs a 2D convolution followed by activation and batch normalization
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) # Applies 2D convolution
        self.activation = nn.Tanh() # non-linear activation function
        self.batchnorm = nn.BatchNorm2d(out_channels) # Applies batch normalization to the output of the convolution -> improve training by normalizing layer inputs

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batchnorm(x)
        return x

# conv_transpose_block class: defines the transpose convolution block used in the decoder, applying transpose convolution, activation, and batch normalization
class conv_transpose_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv_transpose_block, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding) # Applies deconvolution for upsampling
        self.activation = nn.LeakyReLU() # LeakyReLU activation allows a small gradient when the unit is not active
        self.batchnorm = nn.BatchNorm2d(out_channels) # Normalizes the output after each transpose convolution

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.activation(x)
        x = self.batchnorm(x)
        return x

# VAE class: defines the architecture of the autoencoder (encoder, reparameterization trick and decoder)
class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout_rate):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate # Dropout rate for regularization and to avoid overfitting
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)

        # Encoder: maps input data to a latent representation
        self.cnn_encoder = nn.Sequential(
            # 3 blocks of convolutions: convolutional layers reduce the spatial dimensions of the input image
            conv_block(3, 16, 4, 2, 1),
            conv_block(16, 32, 3, 2, 1),
            conv_block(32, 64, 2, 2, 1),
            conv_block(64, 128, 2, 1, 1),
            nn.MaxPool2d(2, 2), # Max pooling reduces the dimensionality
            conv_block(128, 256, 2, 1, 1),
            nn.MaxPool2d(2, 1), 
        )
        self.fc_encoder = nn.Sequential(
            self.dropout,
            nn.Linear(256*9*9, self.hidden_dim), # Fully connected layer that flattens the input and connects to the hidden dimension
            self.activation,
        )
        # Fully connected layer that maps the hidden representation to the mean (mu) of the latent distribution
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            self.activation,
            nn.Linear(self.hidden_dim//2, latent_dim),
        )
        # Fully connected layer that maps the hidden representation to the log variance (logvar) of the latent distribution
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            self.activation,
            nn.Linear(self.hidden_dim//2, latent_dim),
        )

        # Decoder: reconstructs the image from the latent space
        self.cnn_decoder = nn.Sequential(
            # Transpose Convolutions: layers that increase the spatial dimensions, gradually reconstructing the image
            conv_transpose_block(self.hidden_dim, 128, 4, 1, 0),
            conv_transpose_block(128, 64, 4, 2, 1),
            conv_transpose_block(64, 32, 4, 2, 1),
            conv_transpose_block(32, 16, 4, 2, 1),
            conv_transpose_block(16, 8, 4, 2, 1),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),
            nn.Sigmoid(), # Activation function that outputs values between 0 and 1
        )
        
        # Fully connected layers to expand the latent vector before feeding into the transpose convolution layers
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
        )

    def encode(self, x):
        # Convolutional layers
        x = self.cnn_encoder(x)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    # Reparameterization trick: samples from the latent distribution using the mean and log variance
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # Random noise
        return mu + eps*std

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(z.size(0), self.hidden_dim, 1, 1)
        return self.cnn_decoder(z)

    # Forward pass: defines the forward computation through the VAE (encoding, reparameterization and decoding)
    def forward(self, x):
        mu, logvar = self.encode(x) # encode(x): encodes the input image to latent parameters (mu and logvar)
        z = self.reparameterize(mu, logvar) # reparameterize(mu, logvar): samples a latent vector z from the distribution
        return self.decode(z), mu, logvar # decode(z): reconstructs the image from the latent vector