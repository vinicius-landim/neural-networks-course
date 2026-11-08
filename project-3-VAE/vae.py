import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.utils import make_grid

CLASSES = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
           "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(device, root="data", seed=0):
    train = datasets.FashionMNIST(root, train=True, download=True)
    test = datasets.FashionMNIST(root, train=False, download=True)

    # (N, 28, 28) inteiros 0-255  ->  (N, 1, 28, 28) float em [0, 1]
    x_full = train.data.float().unsqueeze(1) / 255.0
    y_full = train.targets
    x_test = test.data.float().unsqueeze(1) / 255.0
    y_test = test.targets

    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=10_000, stratify=y_full, random_state=seed)

    return {
        "x_train": x_train.to(device), "y_train": y_train.to(device),
        "x_val": x_val.to(device), "y_val": y_val.to(device),
        "x_test": x_test.to(device), "y_test": y_test.to(device),
    }

class VAE(nn.Module):
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps,  eps ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

class MLPPadraoVAE(VAE):
    """784 -> 256 -> d"""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

class MLPProfundaVAE(VAE):
    """784 -> 512 -> 256 -> d"""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )


class ConvVAE(VAE):
    """1x28x28 -> 32x14x14 -> 64x7x7 -> 256 -> d"""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # 32 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 7 x 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 1 x 28 x 28
            nn.Sigmoid(),
        )

MODELS = {
    "mlp_padrao": MLPPadraoVAE,
    "mlp_profunda": MLPProfundaVAE,
    "conv": ConvVAE,
}

def build_model(name, latent_dim):
    return MODELS[name](latent_dim)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def vae_loss(x, x_hat, mu, logvar):
    """loss = ||x - x_hat||^2 + KL(N(mu, sigma^2) || N(0, I)), média por imagem."""
    batch_size = x.size(0)
    recon = ((x_hat - x) ** 2).sum() / batch_size
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / batch_size
    return recon + kl, recon, kl

@torch.no_grad()
def evaluate(model, x, batch_size=1000):
    """Loss média (total, reconstrução, KL) em um conjunto de dados."""
    model.eval()
    total, recon, kl = 0.0, 0.0, 0.0
    for xb in x.split(batch_size):
        x_hat, mu, logvar = model(xb)
        l, r, k = vae_loss(xb, x_hat, mu, logvar)
        total += l.item() * len(xb)
        recon += r.item() * len(xb)
        kl += k.item() * len(xb)
    n = len(x)
    return total / n, recon / n, kl / n


@torch.no_grad()
def encode_mu(model, x, batch_size=1000):
    """Média mu do encoder para todo o conjunto (usada nos gráficos do espaço latente)."""
    model.eval()
    mus = [model.encode(xb)[0] for xb in x.split(batch_size)]
    return torch.cat(mus).cpu().numpy()

def manifold_grid(n, device, lim=3.0):
    """Grade n x n de pontos (z1, z2) em [-lim, lim]; z2 decresce de cima para baixo."""
    values = torch.linspace(-lim, lim, n)
    points = [[z1, z2] for z2 in values.flip(0) for z1 in values]
    return torch.tensor(points, device=device)


@torch.no_grad()
def log_images(writer, model, epoch, x_fixed, z_fixed):
    model.eval()

    #reconstruções: linha de cima = originais, linha de baixo = reconstruções
    mu, _ = model.encode(x_fixed)
    recon = model.decode(mu)
    writer.add_image("reconstrucoes", make_grid(torch.cat([x_fixed, recon]), nrow=len(x_fixed)), epoch)

    #gerações a partir dos mesmos z ~ N(0, I) em todas as épocas
    writer.add_image("geracoes", make_grid(model.decode(z_fixed), nrow=8), epoch)

    #manifold (apenas latente 2D)
    if model.latent_dim == 2:
        grid = model.decode(manifold_grid(15, x_fixed.device))
        writer.add_image("manifold_2d", make_grid(grid, nrow=15), epoch)

def train_model(model, data, writer=None, epochs=50, batch_size=128, lr=1e-3, patience=5):
    """Treina com Adam e early stopping pela loss de validação.

    Retorna o modelo com os pesos da melhor época e um dicionário com o resumo.
    """
    device = data["x_train"].device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(data["x_train"]), batch_size=batch_size, shuffle=True)

    #entradas fixas para acompanhar a evolução das imagens no TensorBoard
    x_fixed = data["x_val"][:16]
    z_fixed = torch.randn(64, model.latent_dim, device=device)
    if writer is not None:
        log_images(writer, model, 0, x_fixed, z_fixed)

    best_val, best_epoch, best_state, best_parts = float("inf"), 0, None, None
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, recon, kl, n = 0.0, 0.0, 0.0, 0
        for (xb,) in loader:
            x_hat, mu, logvar = model(xb)
            loss, r, k = vae_loss(xb, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(xb)
            recon += r.item() * len(xb)
            kl += k.item() * len(xb)
            n += len(xb)
        train_loss = (total / n, recon / n, kl / n)

        val_loss = evaluate(model, data["x_val"])
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if writer is not None:
            for i, name in enumerate(["total", "reconstrucao", "kl"]):
                writer.add_scalar(f"{name}/treino", train_loss[i], epoch)
                writer.add_scalar(f"{name}/validacao", val_loss[i], epoch)
            log_images(writer, model, epoch, x_fixed, z_fixed)

        print(f"  época {epoch:3d} | treino {train_loss[0]:8.3f} | val {val_loss[0]:8.3f} "
              f"(recon {val_loss[1]:.3f}, KL {val_loss[2]:.3f})")

        if val_loss[0] < best_val:
            best_val, best_epoch, best_parts = val_loss[0], epoch, val_loss
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  early stopping (melhor época: {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "val_total": best_parts[0],
        "val_recon": best_parts[1],
        "val_kl": best_parts[2],
        "train_time_s": time.time() - start,
        "history": history,
    }