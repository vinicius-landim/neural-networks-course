import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from vae import CLASSES, build_model, encode_mu, evaluate, get_device, load_data, manifold_grid

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948", "#8c6d46", "#5f6b73"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})

def scatter_labels(z, y, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for c in range(10):
        m = y == c
        ax.scatter(z[m, 0], z[m, 1], s=3, alpha=0.6, color=COLORS[c], label=CLASSES[c], rasterized=True)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend(markerscale=4, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(path)
    plt.close(fig)


def scatter_per_class(z, y, title, path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        m = y == c
        ax.scatter(z[~m, 0], z[~m, 1], s=1, color="#d6d4cc", rasterized=True)
        ax.scatter(z[m, 0], z[m, 1], s=2, alpha=0.7, color=COLORS[c], rasterized=True)
        ax.set_title(CLASSES[c], fontsize=10)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def image_grid(imgs, nrow):
    imgs = imgs.squeeze(1)
    n, h, w = imgs.shape
    ncol = int(np.ceil(n / nrow))
    out = np.zeros((ncol * (h + 2), nrow * (w + 2)))  # 0 = branco no mapa gray_r
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        out[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = im
    return out


@torch.no_grad()
def plot_manifold(model, path, n=20, lim=3.0):
    imgs = model.decode(manifold_grid(n, next(model.parameters()).device, lim)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_grid(imgs, n), cmap="gray_r", extent=[-lim, lim, -lim, lim])
    ax.set(title="Manifold do latente 2D (decodificação de uma grade em z)", xlabel="z₁", ylabel="z₂")
    ax.grid(False)
    fig.savefig(path)
    plt.close(fig)


def plot_pca_variance(ratio, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(1, len(ratio) + 1)
    bars = ax.bar(k, ratio * 100, width=0.6, color=["#2a78d6"] * 2 + ["#9ec5f4"] * (len(ratio) - 2))
    for b, r in zip(bars, ratio):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{r * 100:.1f}%",
                ha="center", fontsize=9, color=INK)
    ax.set(title=f"Variância explicada por componente (PC1 + PC2 = {ratio[:2].sum() * 100:.1f}%)",
           xlabel="Componente principal", ylabel="Variância explicada (%)", xticks=k,
           ylim=(0, ratio.max() * 100 * 1.15))
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_val_curves(out, latents, path):
    names = ["mlp_rasa", "mlp_profunda", "conv"]
    fig, axes = plt.subplots(1, len(latents), figsize=(6 * len(latents), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, latents):
        for i, name in enumerate(names):
            hist_path = out / "models" / f"{name}_d{d}_history.json"
            if not hist_path.exists():
                continue
            hist = json.load(open(hist_path))
            ax.plot([h["epoch"] for h in hist], [h["val"][0] for h in hist],
                    lw=2, color=COLORS[i], label=name)
        ax.set(title=f"Latente {d}D: loss de validação", xlabel="Época", ylabel="Loss (recon + KL)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(models, x, y, path):
    # uma imagem de cada classe do teste
    idx = [int(torch.nonzero(y == c)[0]) for c in range(10)]
    xs = x[idx]
    rows = [("Original", xs.cpu().numpy())]
    for d, m in models.items():
        mu, _ = m.encode(xs)
        rows.append((f"Recon. {d}D", m.decode(mu).cpu().numpy()))
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.25 * len(rows) + 0.4))
    for ax, (label, imgs) in zip(axes, rows):
        ax.imshow(image_grid(imgs, 10), cmap="gray_r")
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_title("Reconstruções (uma imagem de cada classe do teste)")
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def plot_samples(models, path, n=64):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5.3))
    axes = np.atleast_1d(axes)
    for ax, (d, m) in zip(axes, models.items()):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(n, d, generator=g).to(next(m.parameters()).device)
        ax.imshow(image_grid(m.decode(z).cpu().numpy(), 8), cmap="gray_r")
        ax.set_title(f"Gerações do latente {d}D (z ~ N(0, I))")
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.out)

    device = get_device()
    data = load_data(device, seed=args.seed)
    x_test, y_test = data["x_test"], data["y_test"]
    y = y_test.cpu().numpy()

    best = json.load(open(out / "best.json"))
    rows = list(csv.DictReader(open(out / "results.csv")))
    latents = sorted(int(d) for d in best)

    models = {}
    for d in latents:
        ckpt = torch.load(out / "models" / f"{best[str(d)]}_d{d}.pt", map_location=device)
        m = build_model(ckpt["name"], d).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models[d] = m

    lines = ["RESULTADOS: VAE no Fashion-MNIST", "=" * 60, "",
             "1) Seleção da topologia (loss de validação = recon + KL)", "-" * 60,
             f"{'latente':>7} {'modelo':>13} {'params':>10} {'val total':>10} {'recon':>9} {'KL':>7}"]
    for r in rows:
        mark = "  <- melhor" if best[r["latent_dim"]] == r["modelo"] else ""
        lines.append(f"{r['latent_dim']:>7} {r['modelo']:>13} {int(r['parametros']):>10,} "
                     f"{float(r['val_total']):>10.3f} {float(r['val_recon']):>9.3f} {float(r['val_kl']):>7.3f}{mark}")
    lines.append("")
    for d, m in models.items():
        t = evaluate(m, x_test)
        lines.append(f"Teste, melhor {d}D ({best[str(d)]}): total {t[0]:.3f} | recon {t[1]:.3f} | KL {t[2]:.3f}")
    lines.append("")

    plot_val_curves(out, latents, out / "curvas_validacao.png")
    plot_reconstructions(models, x_test, y_test, out / "reconstrucoes.png")
    plot_samples(models, out / "amostras.png")

    # Representações latentes do conjunto de teste (média mu, sem amostragem)
    reps = {}
    if 2 in models:
        z2 = encode_mu(models[2], x_test)
        reps["Latente 2D"] = z2
        scatter_labels(z2, y, f"Espaço latente 2D ({best['2']}): μ do conjunto de teste",
                       "z₁", "z₂", out / "latente_2d.png")
        scatter_per_class(z2, y, "Espaço latente 2D: cada classe destacada", out / "latente_2d_por_classe.png")
        plot_manifold(models[2], out / "manifold_2d.png")

    if 8 in models:
        z8 = encode_mu(models[8], x_test)
        # PCA com todos os componentes, para obter todos os autovalores
        pca = PCA().fit(z8)
        proj = pca.transform(z8)[:, :2] # projeção nos 2 primeiros componentes
        eigvals = pca.explained_variance_ # autovalores da matriz de covariância
        ratio = pca.explained_variance_ratio_ # fração da variância por componente
        reps["Latente 8D"] = z8
        reps["PCA 2D do latente 8D"] = proj
        scatter_labels(proj, y, f"Projeção PCA do latente 8D ({best['8']}): {ratio[:2].sum() * 100:.1f}% da variância",
                       f"PC1 ({ratio[0] * 100:.1f}%)", f"PC2 ({ratio[1] * 100:.1f}%)", out / "pca_8d.png")
        scatter_per_class(proj, y, "Projeção PCA do latente 8D: cada classe destacada", out / "pca_8d_por_classe.png")
        plot_pca_variance(ratio, out / "pca_variancia.png")

        lines += ["2) PCA do latente 8D (autovalores da matriz de covariância)", "-" * 60,
                  f"{'PC':>4} {'autovalor':>10} {'variância':>10} {'acumulada':>10}"]
        for i, (lam, r, c) in enumerate(zip(eigvals, ratio, np.cumsum(ratio)), 1):
            lines.append(f"{i:>4} {lam:>10.4f} {r * 100:>9.2f}% {c * 100:>9.2f}%")
        lines += ["", f"A projeção em 2D (PC1 + PC2) ilustra {ratio[:2].sum() * 100:.1f}% da variância do latente 8D.",
                  "(PCA calculado com sklearn.decomposition.PCA.)", ""]

    # Clusters e separação dos rótulos: silhouette usando os rótulos verdadeiros
    lines += ["3) Clusters e separação dos rótulos (silhouette com os rótulos reais)", "-" * 60,
              "Silhouette varia de -1 a 1: perto de 1 = classes compactas e separadas;",
              "perto de 0 = classes sobrepostas; negativo = pontos mais perto de outra classe.", ""]
    per_class = {}
    for name, z in reps.items():
        s = silhouette_samples(z, y)
        per_class[name] = np.array([s[y == c].mean() for c in range(10)])
        lines.append(f"{name:<24} silhouette médio = {s.mean():.3f}")
    lines.append("")
    header = f"{'classe':<10}" + "".join(f"{n:>24}" for n in per_class)
    lines += ["Silhouette médio por classe:", header]
    for c in range(10):
        lines.append(f"{CLASSES[c]:<10}" + "".join(f"{v[c]:>24.3f}" for v in per_class.values()))
    lines.append("")
    for name, v in per_class.items():
        order = np.argsort(v)[::-1]
        lines.append(f"{name}: mais separadas = {', '.join(CLASSES[i] for i in order[:3])}; "
                     f"mais misturadas = {', '.join(CLASSES[i] for i in order[-3:][::-1])}")

    (out / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFiguras e results.txt salvos em {out}/")


if __name__ == "__main__":
    main()