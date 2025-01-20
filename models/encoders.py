import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        log_var = self.fc2_log_var(h)
        return mu, log_var

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_reconstructed = self.fc2(h)
        return x_reconstructed

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, batch_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.batch_size = batch_size
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss_function(self, x, x_reconstructed, mu, log_var):
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        # KL Divergence loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return mse_loss + kld_loss

# large molecule encoder
class LargeMoleculeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LargeMoleculeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2*hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        return x

# small molecule encoder
class SmallMoleculeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SmallMoleculeEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2*hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        return x