import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchsom.core import SOM

RANDOM_SEED = 42

def load_image(image_path):
    """Carrega uma imagem RGB normalizada para [0, 1]."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Não foi possível carregar: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    return image


def train_som(image, side, device):
    """Treina o SOM sobre os pixels da imagem."""

    pixels = image.reshape(-1, 3)
    pixels = torch.from_numpy(pixels)

    torch.manual_seed(RANDOM_SEED)

    som = SOM(
        x=side,
        y=side,
        num_features=3,
        sigma=side / 2,
        learning_rate=0.5,
        epochs=30,
        batch_size=512,
        neighborhood_order=1,
        topology="rectangular",
        distance_function="euclidean",
        neighborhood_function="gaussian",
        lr_decay_function="lr_inverse_decay_to_zero",
        sigma_decay_function="sig_inverse_decay_to_one",
        search_backend="torch",
        random_seed=RANDOM_SEED,
        device=device,
    )

    som.initialize_weights(
        data=pixels,
        mode="random",
    )

    QE, TE = som.fit(data=pixels)

    #fit devolve o TE em porcentagem->conversão para fração, de mesma escala que som.topographic_error().
    TE = [t / 100.0 for t in TE]

    return som, QE, TE


@torch.no_grad()
def final_topographic_error(som, image, chunk=100_000):
    """TE final (fração 0-1) sobre todos os pixels, com os pesos finais."""

    pixels = torch.from_numpy(image.reshape(-1, 3))

    total = 0.0

    for i in range(0, len(pixels), chunk):
        batch = pixels[i:i + chunk]
        total += som.topographic_error(batch) * len(batch)

    return total / len(pixels)


@torch.no_grad()
def reconstruct(image, palette, chunk=200_000):
    """Reconstrói a imagem utilizando a paleta do SOM."""

    height, width, _ = image.shape

    pixels = torch.from_numpy(
        image.reshape(-1, 3)
    ).float()

    palette = palette.float()

    bmu = torch.empty(
        len(pixels),
        dtype=torch.long,
    )

    distances = torch.empty(
        len(pixels),
    )

    for i in range(0, len(pixels), chunk):

        d = torch.cdist(
            pixels[i:i + chunk],
            palette,
        )

        minimum = d.min(dim=1)

        bmu[i:i + chunk] = minimum.indices
        distances[i:i + chunk] = minimum.values

    reconstructed = palette[bmu].reshape(
        height,
        width,
        3,
    ).numpy()

    return reconstructed, bmu, distances


def calculate_metrics(image, reconstructed, distances):
    """Calcula QE, MSE e PSNR."""

    mse = float(
        ((image - reconstructed) ** 2).mean()
    )
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    qe = float(distances.mean())

    return qe, mse, psnr


def run_experiment(image_path, side, device):
    """Executa o experimento completo para uma imagem."""

    image = load_image(image_path)
    t0 = time.time()

    som, QE, TE = train_som(
        image,
        side,
        device,
    )

    train_time = time.time() - t0

    weights = som.weights.detach().cpu()

    palette = (
        weights
        .reshape(-1, 3)
        .clamp(0, 1)
    )

    grid = (
        palette
        .reshape(side, side, 3)
        .numpy()
    )

    reconstructed, bmu, distances = reconstruct(
        image,
        palette,
    )

    qe, mse, psnr = calculate_metrics(
        image,
        reconstructed,
        distances,
    )

    te = final_topographic_error(som, image)

    dead_units = int(
        (torch.bincount(bmu, minlength=len(palette)) == 0).sum()
    )

    return {
        "image": image,
        "reconstructed": reconstructed,
        "palette": palette,
        "grid": grid,
        "bmu": bmu,
        "distances": distances,
        "QE": qe,
        "TE": te,
        "MSE": mse,
        "PSNR": psnr,
        "dead_units": dead_units,
        "train_time": train_time,
        "QE_training": QE,
        "TE_training": TE,
    }