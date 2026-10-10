import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import PredictionErrorDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import json

def data_preprocessing(csv_path, batch_size):
    dataset = pd.read_csv(csv_path)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values

    X_test, X_temp, y_test, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_set = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, X_val_t, y_val_t, X_test_t, y_test_t, scaler


class MLP(nn.Module):
    def __init__(self, fc1_size, fc2_size, fc3_size, p1=0.0, p2=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, fc1_size),
            nn.ReLU(),
            nn.Dropout(p1),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Dropout(p2),
            nn.Linear(fc2_size, fc3_size),
            nn.ReLU(),
            nn.Linear(fc3_size, 1)
        )

    def forward(self, X):
        return self.layers(X)


def train(model, device, train_loader, X_val, y_val, criterion, optimizer, l1_lambda=0.0, n_epochs=100):
    train_loss_hist = []
    val_loss_hist = []
    train_mae_hist = []
    val_mae_hist = []

    mae_criterion = nn.L1Loss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(n_epochs):
        model.train()

        batch_losses = []
        batch_maes = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            mse_loss = criterion(y_pred, y_batch)

            loss = mse_loss

            if l1_lambda > 0.0:
                l1_norm = sum(
                    param.abs().sum()
                    for param in model.parameters()
                )
                loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()

            batch_losses.append(mse_loss.item())

            with torch.no_grad():
                mae = mae_criterion(y_pred, y_batch)
                batch_maes.append(mae.item())

        avg_train_loss = np.mean(batch_losses)
        avg_train_mae = np.mean(batch_maes)

        train_loss_hist.append(avg_train_loss)
        train_mae_hist.append(avg_train_mae)

        model.eval()

        with torch.no_grad():
            y_val_pred = model(X_val)

            val_loss = criterion(y_val_pred, y_val).item()
            val_mae = mae_criterion(y_val_pred, y_val).item()

        val_loss_hist.append(val_loss)
        val_mae_hist.append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch == 0 or (epoch + 1) % 500 == 0 or (epoch + 1) == n_epochs):
            print(f"Epoch {epoch + 1:5d}/{n_epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    idx = best_epoch - 1

    best_metrics = {
        "epoch": best_epoch,
        "train": {
            "MSE": float(train_loss_hist[idx]),
            "RMSE": float(np.sqrt(train_loss_hist[idx])),
            "MAE": float(train_mae_hist[idx]),
        },
        "validation": {
            "MSE": float(val_loss_hist[idx]),
            "RMSE": float(np.sqrt(val_loss_hist[idx])),
            "MAE": float(val_mae_hist[idx]),
        },
    }

    print("\n" + "=" * 60)
    print(" TRAINING SUMMARY")
    print("=" * 60)
    print(f" Best Epoch : {best_metrics['epoch']}")
    print("-" * 60)
    print(
        f" Train | "
        f"MSE: {best_metrics['train']['MSE']:.4f} | "
        f"RMSE: {best_metrics['train']['RMSE']:.4f} | "
        f"MAE: {best_metrics['train']['MAE']:.4f}"
    )
    print(
        f" Val   | "
        f"MSE: {best_metrics['validation']['MSE']:.4f} | "
        f"RMSE: {best_metrics['validation']['RMSE']:.4f} | "
        f"MAE: {best_metrics['validation']['MAE']:.4f}"
    )
    print("=" * 60 + "\n")

    return (
        train_loss_hist,
        val_loss_hist,
        train_mae_hist,
        val_mae_hist,
        best_metrics,
    )


def eval(model, X_test, y_test, criterion, device):
    model.eval()

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        y_pred = model(X_test)

        mse = criterion(y_pred, y_test).item()

        mae_criterion = nn.L1Loss()
        mae = mae_criterion(y_pred, y_test).item()

    rmse = np.sqrt(mse)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "y_true": y_test.cpu().numpy().flatten(),
        "y_pred": y_pred.cpu().numpy().flatten()
    }


def plot_training_history(train_loss, val_loss, train_mae, val_mae, title="Treinamento", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_loss, label="Treino (MSE)", linewidth=1)
    ax1.plot(val_loss, label="Validação (MSE)", linewidth=1)

    ax1.set_title("Evolução da Loss (MSE)", fontsize=12)
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Erro Quadrático Médio")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(train_mae, label="Treino (MAE)", linewidth=1)
    ax2.plot(val_mae, label="Validação (MAE)", linewidth=1)

    ax2.set_title("Evolução da Métrica Auxiliar (MAE)", fontsize=12)
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Erro Absoluto Médio")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)



def plot_parity_and_residuals(y_true, y_pred, title="Avaliação", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, 
        y_pred=y_pred, 
        kind="actual_vs_predicted", 
        ax=ax1,
        scatter_kwargs={"alpha": 0.7, "edgecolors": "k"}
    )
    ax1.set_title("Gráfico de Paridade")

    PredictionErrorDisplay.from_predictions(
        y_true=y_true, 
        y_pred=y_pred, 
        kind="residual_vs_predicted", 
        ax=ax2,
        scatter_kwargs={"alpha": 0.7, "edgecolors": "k"}
    )
    ax2.set_title("Gráfico de Resíduos")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    FC1_SIZE = 64
    FC2_SIZE = 128
    FC3_SIZE = 64

    EPOCHS = 5000
    BATCH_SIZE = 5

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    train_loader, X_val, y_val, X_test, y_test, scaler = data_preprocessing(
        "../data/dataset_projeto1.csv",
        batch_size=BATCH_SIZE
    )

    print("All data loaded")
    print(f"Device: {device}")
    print(f"Architecture: 1 -> {FC1_SIZE} -> {FC2_SIZE} -> {FC3_SIZE} -> 1")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    models = {
        "Baseline": {"p1": 0.0, "p2": 0.0, "momentum": 0.0, "l1": 0.0, "l2": 0.0},
        "Baseline + Momentum": {"p1": 0.0, "p2": 0.0, "momentum": 0.9, "l1": 1e-5, "l2": 0.0},
        "Baseline + L1": {"p1": 0.0, "p2": 0.0, "momentum": 0.0, "l1": 1e-5, "l2": 0.0},
        "Baseline + L2": {"p1": 0.0, "p2": 0.0, "momentum": 0.0, "l1": 0.0, "l2": 1e-6},
        "Baseline + Dropout": {"p1": 0.05, "p2": 0.1, "momentum": 0.0, "l1": 0.0, "l2": 0.0},
        "Modelo Combinado": {"p1": 0.0, "p2": 0.0, "momentum": 0.9, "l1": 0.0, "l2": 1e-6},
    }

    all_results = {}

    for name, params in models.items():
        print("\n" + "=" * 60)
        print(f" TRAINING MODEL: {name}")
        print("=" * 60)

        torch.manual_seed(42)

        model = MLP(
            fc1_size=FC1_SIZE,
            fc2_size=FC2_SIZE,
            fc3_size=FC3_SIZE,
            p1=params["p1"],
            p2=params["p2"],
        ).to(device)

        loss_fn = nn.MSELoss()

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=params["momentum"],
            weight_decay=params["l2"],
        )

        train_loss, val_loss, train_mae, val_mae, metrics= train(
            model=model,
            device=device,
            train_loader=train_loader,
            X_val=X_val,
            y_val=y_val,
            criterion=loss_fn,
            optimizer=optimizer,
            l1_lambda=params["l1"],
            n_epochs=EPOCHS,
        )

        test_metrics = eval(model, X_test, y_test, loss_fn, device)

        print("=" * 60)
        print(" TEST SUMMARY")
        print("=" * 60)
        print(
            f" Test | "
            f"MSE: {test_metrics['MSE']:.4f} | "
            f"RMSE: {test_metrics['RMSE']:.4f} | "
            f"MAE: {test_metrics['MAE']:.4f}"
        )
        print("=" * 60 + "\n")
        
        safe_name = name.replace(" ", "_").replace("+", "e").lower()
        
        if name in ["Baseline", "Modelo Combinado"]:
            y_true = test_metrics.get("y_true")
            y_pred = test_metrics.get("y_pred")
            
            plot_parity_and_residuals(
                y_true=y_true, 
                y_pred=y_pred, 
                title=f"Avaliação (Dados de Teste) - {name}", 
                save_path=f"outputs/{safe_name}_evaluation.png"
            )

        test_metrics.pop("y_true", None)
        test_metrics.pop("y_pred", None)

        all_results[name] = {
            "configuration": {
                "architecture": [1, FC1_SIZE, FC2_SIZE, FC3_SIZE, 1],
                "p1": params["p1"],
                "p2": params["p2"],
                "momentum": params["momentum"],
                "l1_lambda": params["l1"],
                "weight_decay": params["l2"],
                "learning_rate": 0.01,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "seed": 42,
            },
            "best_epoch": metrics["epoch"],
            "train": metrics["train"],
            "validation": metrics["validation"],
            "test": test_metrics,
        }

        plot_training_history(
            train_loss,
            val_loss,
            train_mae,
            val_mae,
            title=f"Treinamento - {name}",
            save_path=f"outputs/{safe_name}_history.png",
        )

    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/final_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)

    for name, result in all_results.items():
        print(
            f"{name:<22} | "
            f"Val MSE: {result['validation']['MSE']:.4f} | "
            f"Test MSE: {result['test']['MSE']:.4f}"
        )

    print("=" * 60)
    print(f"Results saved to: {json_path}")