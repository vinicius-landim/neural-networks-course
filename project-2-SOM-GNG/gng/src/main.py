import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gng import GrowingNeuralGas

DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
MAX_NODES_LIST = [16, 64, 256]
EPOCHS = 1
TRAIN_FRACTION = 1.0
LOG_EVERY = 500
N_EVAL = 5000
RANDOM_SEED = 42

def load_image(image_path):
    """Carrega uma imagem RGB normalizada para [0, 1]."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Não foi possível carregar: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _top2(W, pixels, chunk=20_000):
    """1º e 2º BMU (índices em W) e distância ao 1º, em blocos."""
    pixels = np.asarray(pixels, dtype=np.float32)
    w2 = (W ** 2).sum(1)

    n = len(pixels)
    b1 = np.empty(n, dtype=np.int64)
    b2 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n, dtype=np.float32)

    for i in range(0, n, chunk):
        x = pixels[i:i + chunk]
        d2 = (x ** 2).sum(1)[:, None] - 2.0 * (x @ W.T) + w2[None, :]
        np.maximum(d2, 0.0, out=d2)

        part = np.argpartition(d2, 1, axis=1)[:, :2]
        r = np.arange(len(x))

        b1[i:i + chunk] = part[:, 0]
        b2[i:i + chunk] = part[:, 1]
        d1[i:i + chunk] = np.sqrt(d2[r, part[:, 0]])

    return b1, b2, d1


def evaluate(gng, pixels, return_quantized=True):
    """QE, TE (1º e 2º BMU ligados por aresta), neurônios mortos e pixels quantizados."""
    act = np.flatnonzero(gng.is_active)
    W = gng.nodes[act]
    E = gng.edges[np.ix_(act, act)]

    b1, b2, d1 = _top2(W, pixels)

    res = {
        "QE": float(d1.mean()),
        "TE": float((~E[b1, b2]).mean()),
        "dead_units": int((np.bincount(b1, minlength=len(act)) == 0).sum()),
        "palette": W.copy(),
    }
    if return_quantized:
        res["quantized"] = W[b1]
    return res


class LoggedPixels:
    """
    Envolve o array de pixels para registrar QE/TE durante o treino
    sem alterar a classe GNG.

    A GNG lê um pixel por iteração (pixels[idx]). Este objeto conta esses
    acessos e, a cada `every` iterações, mede a rede em uma amostra fixa.
    """

    def __init__(self, pixels, gng, sample, every):
        self.pixels = pixels
        self.gng = gng
        self.sample = sample
        self.every = every
        self.count = 0
        self.history = {"iter": [], "epoch": [], "nodes": [], "QE": [], "TE": []}

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        self.count += 1

        completed = self.count - 3
        if completed > 0 and completed % self.every == 0:
            self._log(completed)

        return self.pixels[idx]

    def _log(self, iteration):
        if int(self.gng.is_active.sum()) < 2:
            return
        res = evaluate(self.gng, self.sample, return_quantized=False)
        self.history["iter"].append(iteration)
        self.history["epoch"].append(iteration / len(self.pixels))  # época (fracionária)
        self.history["nodes"].append(int(self.gng.is_active.sum()))
        self.history["QE"].append(res["QE"])
        self.history["TE"].append(res["TE"])

    def finish(self):
        """Registra o ponto final do treino."""
        completed = self.count - 2
        if completed > 0 and (not self.history["iter"] or self.history["iter"][-1] != completed):
            self._log(completed)


def save_image(image, path):
    """Salva uma imagem RGB [0,1]."""
    image = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def palette_to_grid(palette):
    """Paleta linear -> grid quadrado (ordenada por luminância, só para exibir)."""
    lum = palette @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    palette = palette[np.argsort(lum)]

    k = len(palette)
    side = int(np.ceil(np.sqrt(k)))
    square = np.zeros((side * side, 3), dtype=np.float32)
    square[:k] = palette
    return square.reshape(side, side, 3)


def save_result_images(result, output_dir, k):
    """Salva reconstrução e paleta da GNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(result["reconstructed"], output_dir / f"reconstructed_K{k}.png")

    grid = result["grid"]
    side = grid.shape[0]
    grid = cv2.resize(grid, (side * 32, side * 32), interpolation=cv2.INTER_NEAREST)
    save_image(grid, output_dir / f"palette_K{k}.png")


def to_percent(values):
    """Converte uma curva de fração (0-1) para porcentagem (0-100)."""
    return [100.0 * float(v) for v in values]


def save_training_curves(result, output_dir, k, image_name=""):
    """Curvas de QE (com nº de nós) e TE (%) por época de um único treino."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h = result["history"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    ax[0].plot(h["epoch"], h["QE"])
    ax[0].set_title("Erro de quantização")
    ax[0].set_ylabel("QE")
    twin = ax[0].twinx()
    twin.plot(h["epoch"], h["nodes"], color="gray", ls="--")
    twin.set_ylabel("nº de nós")

    te = to_percent(h["TE"])  # fração -> %
    ax[1].plot(h["epoch"], te, color="tab:orange")
    ax[1].set_title("Erro topológico")
    ax[1].set_ylabel("TE (%)")
    ax[1].set_ylim(0, max(5.0, max(te) * 1.1))

    for a in ax:
        a.set_xlabel("Época")
        a.set_xlim(0, EPOCHS)
        a.set_xticks(np.linspace(0, EPOCHS, 6))
        a.grid(alpha=0.3)

    fig.suptitle(f"{image_name} | GNG K={k}")
    fig.tight_layout()
    fig.savefig(output_dir / f"training_curves_K{k}.png", dpi=150)
    plt.close(fig)


def save_comparison_curves(histories, output_dir, image_name=""):
    """QE e TE (%) dos diferentes K sobrepostos. histories: {k: history}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    max_te = 0.0

    for k, h in histories.items():
        te = to_percent(h["TE"])  # fração -> %
        ax[0].plot(h["epoch"], h["QE"], label=f"K={k}")
        ax[1].plot(h["epoch"], te, label=f"K={k}")
        max_te = max(max_te, max(te))

    ax[0].set_title("Erro de quantização")
    ax[0].set_ylabel("QE")
    ax[1].set_title("Erro topológico")
    ax[1].set_ylabel("TE (%)")
    ax[1].set_ylim(0, max(5.0, max_te * 1.1))

    for a in ax:
        a.set_xlabel("Época")
        a.set_xlim(0, EPOCHS)
        a.set_xticks(np.linspace(0, EPOCHS, 6))
        a.legend()
        a.grid(alpha=0.3)

    fig.suptitle(f"{image_name} | GNG: comparação entre K")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves_all.png", dpi=150)
    plt.close(fig)


def run_experiment(image_path, max_nodes, epochs=EPOCHS, seed=RANDOM_SEED):
    """Pipeline completo da GNG, no mesmo formato de saída do SOM."""
    image = load_image(image_path)
    pixels = image.reshape(-1, 3)

    # amostra de treino e amostra fixa para as curvas
    rng = np.random.default_rng(seed)
    if TRAIN_FRACTION >= 1.0:
        train = pixels
    else:
        n_train = max(3, int(len(pixels) * TRAIN_FRACTION))
        train = pixels[rng.choice(len(pixels), n_train, replace=False)]
    sample = pixels[rng.choice(len(pixels), min(N_EVAL, len(pixels)), replace=False)]

    np.random.seed(seed)

    gng = GrowingNeuralGas(max_nodes=max_nodes)
    every = LOG_EVERY if LOG_EVERY else len(train)  # None -> um ponto por época
    logged = LoggedPixels(train, gng, sample, every)

    t0 = time.time()
    gng.fit(logged, epochs=epochs)
    elapsed = time.time() - t0
    logged.finish()

    # inferência (todos os pixels) e métricas
    ev = evaluate(gng, pixels)
    reconstructed = ev["quantized"].reshape(image.shape)

    mse = float(((image - reconstructed) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

    return {
        "image": image,
        "reconstructed": reconstructed,
        "grid": palette_to_grid(ev["palette"]),
        "QE": ev["QE"],
        "TE": ev["TE"],
        "MSE": mse,
        "PSNR": psnr,
        "dead_units": ev["dead_units"],
        "colors_count": len(ev["palette"]),
        "train_time": elapsed,
        "history": logged.history,
    }


def get_image_paths(data_dir):
    return sorted(path for path in data_dir.glob("*.jpg") if path.is_file())


def main():
    print("Iniciando pipeline de experimentos GNG...")

    image_paths = get_image_paths(DATA_DIR)
    print(f"{len(image_paths)} imagens encontradas em {DATA_DIR.resolve()}.")

    results = []

    for image_path in image_paths:
        print(f"\n{'=' * 60}")
        print(f"Imagem: {image_path.name}")

        image_results_dir = RESULTS_DIR / image_path.stem
        histories = {}

        for k in MAX_NODES_LIST:
            print(f"\nExecutando GNG com K = {k}...")

            result = run_experiment(image_path, k)

            save_result_images(result, image_results_dir, k)
            save_training_curves(result, image_results_dir, k, image_name=image_path.stem)

            h = result["history"]
            histories[k] = h

            results.append({
                "image": image_path.name,
                "max_nodes_target": k,
                "num_colors_obtained": result["colors_count"],
                "QE": result["QE"],
                "TE": result["TE"],
                "MSE": result["MSE"],
                "PSNR": result["PSNR"],
                "dead_units": result["dead_units"],
                "QE_train_final": h["QE"][-1],
                "TE_train_final": h["TE"][-1],
                "train_time_s": result["train_time"],
            })

            if result["colors_count"] != k:
                print(f"AVISO: a rede terminou com {result['colors_count']} nós (alvo {k}).")

            print(f"Cores obtidas = {result['colors_count']}")
            print(f"QE            = {result['QE']:.4f}")
            print(f"TE            = {result['TE']:.4f}")
            print(f"MSE           = {result['MSE']:.6f}")
            print(f"PSNR          = {result['PSNR']:.2f} dB")
            print(f"Neurônios mortos = {result['dead_units']}")
            print(f"Tempo de treino  = {result['train_time']:.1f} s")

            # salva a cada experimento: uma interrupção não perde o que já rodou
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(RESULTS_DIR / "metrics_gng.csv", index=False)

        save_comparison_curves(histories, image_results_dir, image_name=image_path.stem)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if results:
        pd.DataFrame(results).to_csv(RESULTS_DIR / "metrics_gng.csv", index=False)

    print(f"\nResultados salvos em: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()