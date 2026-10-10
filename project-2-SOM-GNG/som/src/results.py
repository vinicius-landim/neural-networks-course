import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import cv2
from pathlib import Path

def to_list(values):
    """Converte lista, array ou tensor em lista de floats."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return [float(v) for v in values]


def to_percent(values):
    """Converte uma curva de fração (0-1) para porcentagem (0-100)."""
    return [100.0 * v for v in to_list(values)]


def save_image(image, path):
    """Salva uma imagem RGB [0,1]."""
    image = np.clip(np.round(image * 255), 0, 255)
    image = image.astype(np.uint8)

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)

def save_result_images(result, output_dir, side):
    """Salva reconstrução e paleta."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    save_image(
        result["reconstructed"],
        output_dir / f"reconstructed_{side}x{side}.png",
    )

    # apliamento da paleta
    grid = np.clip(np.round(result["grid"] * 255), 0, 255).astype(np.uint8)
    grid = cv2.resize(
        grid,
        (side * 32, side * 32),
        interpolation=cv2.INTER_NEAREST,
    )
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        str(output_dir / f"palette_{side}x{side}.png"),
        grid,
    )


def _te_limit(te_percent):
    """Limite superior do eixo do TE (em %), para não ficar achatado."""
    return max(5.0, max(te_percent) * 1.1)


def save_training_curves(result, output_dir, side, image_name=""):
    """Salva a curva de QE e TE (em %) por época de um único treino."""
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qe = to_list(result["QE_training"])
    te = to_percent(result["TE_training"])
    epochs = range(1, len(qe) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))

    ax[0].plot(epochs, qe)
    ax[0].set_title("Erro de quantização")
    ax[0].set_ylabel("QE")

    ax[1].plot(epochs, te, color="tab:orange")
    ax[1].set_title("Erro topológico")
    ax[1].set_ylabel("TE (%)")
    ax[1].set_ylim(0, _te_limit(te))

    for a in ax:
        a.set_xlabel("Época")
        a.grid(alpha=0.3)

    fig.suptitle(f"{image_name} | SOM {side}x{side}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"training_curves_{side}x{side}.png",
        dpi=150,
    )
    plt.close(fig)


def save_comparison_curves(curves, output_dir, image_name=""):
    """
    Salva QE e TE (em %) dos diferentes grids sobrepostos em um único gráfico.
    curves: dict {side: (qe_curve, te_curve)}, com o TE em fração (0-1).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    max_te = 0.0

    for side, (qe, te) in curves.items():
        qe = to_list(qe)
        te = to_percent(te)

        ax[0].plot(
            range(1, len(qe) + 1),
            qe,
            label=f"{side}x{side}",
        )
        ax[1].plot(
            range(1, len(te) + 1),
            te,
            label=f"{side}x{side}",
        )

        max_te = max(max_te, max(te))

    ax[0].set_title("Erro de quantização")
    ax[0].set_ylabel("QE")

    ax[1].set_title("Erro topológico")
    ax[1].set_ylabel("TE (%)")
    ax[1].set_ylim(0, _te_limit([max_te]))

    for a in ax:
        a.set_xlabel("Época")
        a.legend()
        a.grid(alpha=0.3)

    fig.suptitle(f"{image_name} | SOM: comparação entre grids")
    fig.tight_layout()
    fig.savefig(
        output_dir / "training_curves_all.png",
        dpi=150,
    )
    plt.close(fig)