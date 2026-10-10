from pathlib import Path
import pandas as pd
import torch

from som import run_experiment
from results import (
    save_comparison_curves,
    save_result_images,
    save_training_curves,
    to_list,
)

DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")

SIDES = [4, 8, 16]

def get_image_paths(data_dir):
    return sorted(path for path in data_dir.glob("*.jpg") if path.is_file())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_paths = get_image_paths(DATA_DIR)
    print(f"{len(image_paths)} imagens encontradas.")

    results = []

    for image_path in image_paths:
        print(f"\n{'=' * 60}")
        print(f"Imagem: {image_path.name}")

        image_results_dir = RESULTS_DIR / image_path.stem

        curves = {}

        for side in SIDES:
            print(f"\nExecutando SOM {side}x{side}...")

            result = run_experiment(image_path, side, device)

            save_result_images(result, image_results_dir, side)

            #curva individual do treino
            save_training_curves(
                result, 
                image_results_dir, 
                side, 
                image_name=image_path.stem
            )

            qe_curve = to_list(result["QE_training"])
            te_curve = to_list(result["TE_training"])

            curves[side] = (qe_curve, te_curve)

            results.append({
                "image": image_path.name,
                "side": side,
                "QE": result["QE"],
                "TE": result["TE"],
                "MSE": result["MSE"],
                "PSNR": result["PSNR"],
                "dead_units": result["dead_units"],
                "QE_train_final": qe_curve[-1],
                "TE_train_final": te_curve[-1],
                "train_time_s": result["train_time"],
            })

            print(f"QE   = {result['QE']:.4f}")
            print(f"TE   = {result['TE']:.4f}")
            print(f"MSE  = {result['MSE']:.6f}")
            print(f"PSNR = {result['PSNR']:.2f} dB")
            print(f"Neurônios mortos = {result['dead_units']}")
            print(f"Tempo de treino  = {result['train_time']:.1f} s")

        #gráfico com os três grids sobrepostos
        save_comparison_curves(
            curves, 
            image_results_dir, 
            image_name=image_path.stem
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(results)
    dataframe.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print(f"\nResultados salvos em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()