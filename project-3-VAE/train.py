import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from vae import MODELS, build_model, count_params, get_device, load_data, set_seed, train_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latents", type=int, nargs="+", default=[2, 8])
    p.add_argument("--out", default="outputs")
    p.add_argument("--runs", default="runs")
    args = p.parse_args()

    out, runs = Path(args.out), Path(args.runs)
    (out / "models").mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    data = load_data(device, seed=args.seed)
    print(f"Treino {len(data['x_train'])} | validação {len(data['x_val'])} | teste {len(data['x_test'])}\n")
    print(f"Treinando {args.epochs} épocas completas. Modelo final é o da época com menor loss de validação.\n")

    rows = []
    for d in args.latents:
        for name in MODELS:
            run_name = f"{name}_d{d}"
            print(f"=== {run_name} ===")
            set_seed(args.seed)
            model = build_model(name, d)
            shutil.rmtree(runs / run_name, ignore_errors=True)
            writer = SummaryWriter(runs / run_name)

            model, info = train_model(model, data, writer, epochs=args.epochs,
                                      batch_size=args.batch_size, lr=args.lr, patience=args.epochs)
            writer.close()

            torch.save({"name": name, "latent_dim": d, "state_dict": model.state_dict()},
                       out / "models" / f"{run_name}.pt")
            rows.append({
                "latent_dim": d, "modelo": name, "parametros": count_params(model),
                "melhor_epoca": info["best_epoch"], "epocas_rodadas": info["epochs_run"],
                "val_total": round(info["val_total"], 4), "val_recon": round(info["val_recon"], 4),
                "val_kl": round(info["val_kl"], 4), "tempo_s": round(info["train_time_s"], 1),
            })
            with open(out / "models" / f"{run_name}_history.json", "w") as f:
                json.dump(info["history"], f)

            print(f"Selecionado: época {info['best_epoch']} de {info['epochs_run']} | "
                  f"val total {info['val_total']:.3f} (recon {info['val_recon']:.3f} + "
                  f"KL {info['val_kl']:.3f}) | {info['train_time_s']:.1f} s")
            print()

    #tabela de resultados
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    #melhor candidato por dimensão latente (menor loss de validação)
    best = {}
    for d in args.latents:
        cand = [r for r in rows if r["latent_dim"] == d]
        best[str(d)] = min(cand, key=lambda r: r["val_total"])["modelo"]
    with open(out / "best.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"{'latente':>7} {'modelo':>13} {'params':>9} {'melhor época':>13} {'val total':>10} {'recon':>9} {'KL':>7}")
    for r in rows:
        mark = "  <- melhor" if best[str(r["latent_dim"])] == r["modelo"] else ""
        print(f"{r['latent_dim']:>7} {r['modelo']:>13} {r['parametros']:>9,} {r['melhor_epoca']:>13} "
              f"{r['val_total']:>10.3f} {r['val_recon']:>9.3f} {r['val_kl']:>7.3f}{mark}")
    print(f"\nResultados em {out}/results.csv e {out}/best.json")


if __name__ == "__main__":
    main()