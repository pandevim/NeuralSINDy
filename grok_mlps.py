"""
Phase 1: Train MLPs to "grok" basic mathematical operations.

Grokking = delayed generalization. We train well past overfitting with weight decay,
and the network eventually restructures its weights into an algorithmic circuit
that generalizes perfectly.

We train 4 MLPs: cos(x), sin(x), x+y, x*y
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path


# ─── MLP Architecture ────────────────────────────────────────────────────────

class GrokMLP(nn.Module):
    """Small MLP designed to grok a mathematical function."""

    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, activation="tanh"):
        super().__init__()
        layers = []
        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn)

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Data Generation ─────────────────────────────────────────────────────────

def make_unary_data(func, n_train=500, n_val=200, x_range=(-2 * np.pi, 2 * np.pi)):
    """Generate train/val data for a unary function f(x)."""
    rng = np.random.default_rng(42)

    x_train = rng.uniform(*x_range, size=(n_train, 1)).astype(np.float32)
    y_train = func(x_train).astype(np.float32)

    x_val = rng.uniform(*x_range, size=(n_val, 1)).astype(np.float32)
    y_val = func(x_val).astype(np.float32)

    return (
        torch.from_numpy(x_train), torch.from_numpy(y_train),
        torch.from_numpy(x_val), torch.from_numpy(y_val),
    )


def make_binary_data(func, n_train=1000, n_val=400, x_range=(-5.0, 5.0)):
    """Generate train/val data for a binary function f(x, y)."""
    rng = np.random.default_rng(42)

    xy_train = rng.uniform(*x_range, size=(n_train, 2)).astype(np.float32)
    z_train = func(xy_train[:, 0:1], xy_train[:, 1:2]).astype(np.float32)

    xy_val = rng.uniform(*x_range, size=(n_val, 2)).astype(np.float32)
    z_val = func(xy_val[:, 0:1], xy_val[:, 1:2]).astype(np.float32)

    return (
        torch.from_numpy(xy_train), torch.from_numpy(z_train),
        torch.from_numpy(xy_val), torch.from_numpy(z_val),
    )


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_grok(
    model, x_train, y_train, x_val, y_val,
    epochs=20000, lr=1e-3, weight_decay=1.0,
    log_every=500, device="cpu"
):
    """
    Train with heavy weight decay — the key ingredient for grokking.
    Weight decay forces the network to find a simpler, generalizing solution
    rather than memorizing via large, complex weight patterns.
    """
    model = model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "epoch": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Validate ──
        if epoch % log_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["epoch"].append(epoch)

            print(
                f"  Epoch {epoch:>6d} | "
                f"Train Loss: {loss.item():.6f} | "
                f"Val Loss: {val_loss.item():.6f}"
            )

    return history


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # ── Define the functions to grok ──
    tasks = {
        "cos": {
            "func": lambda x: np.cos(x),
            "input_dim": 1,
            "activation": "tanh",
            "data_fn": lambda f: make_unary_data(f),
            "epochs": 20000,
            "weight_decay": 1.0,
            "lr": 1e-3,
        },
        "sin": {
            "func": lambda x: np.sin(x),
            "input_dim": 1,
            "activation": "tanh",
            "data_fn": lambda f: make_unary_data(f),
            "epochs": 20000,
            "weight_decay": 1.0,
            "lr": 1e-3,
        },
        "add": {
            "func": lambda x, y: x + y,
            "input_dim": 2,
            "activation": "relu",
            "data_fn": lambda f: make_binary_data(f),
            "epochs": 15000,
            "weight_decay": 0.5,
            "lr": 1e-3,
        },
        "mul": {
            "func": lambda x, y: x * y,
            "input_dim": 2,
            "activation": "relu",
            "data_fn": lambda f: make_binary_data(f),
            "epochs": 20000,
            "weight_decay": 0.5,
            "lr": 1e-3,
        },
        "identity": {
            "func": lambda x: x,
            "input_dim": 1,
            "activation": "relu",
            "data_fn": lambda f: make_unary_data(f, x_range=(-5.0, 5.0)),
            "epochs": 10000,
            "weight_decay": 1.0,
            "lr": 1e-3,
        },
    }

    all_histories = {}

    for name, cfg in tasks.items():
        print(f"{'='*60}")
        print(f"  Training MLP_{name}")
        print(f"{'='*60}")

        # Build model
        model = GrokMLP(
            input_dim=cfg["input_dim"],
            hidden_dim=128,
            num_layers=2,
            activation=cfg["activation"],
        )

        # Generate data
        x_train, y_train, x_val, y_val = cfg["data_fn"](cfg["func"])

        # Train
        history = train_grok(
            model, x_train, y_train, x_val, y_val,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
        )

        all_histories[name] = history

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": cfg["input_dim"],
            "activation": cfg["activation"],
            "hidden_dim": 128,
            "num_layers": 2,
        }, save_dir / f"mlp_{name}.pt")

        # Final validation error
        model.eval()
        with torch.no_grad():
            x_val_dev = x_val.to(device)
            y_val_dev = y_val.to(device)
            val_pred = model(x_val_dev)
            final_mse = nn.MSELoss()(val_pred, y_val_dev).item()
            # Relative error
            y_range = (y_val_dev.max() - y_val_dev.min()).item()
            rel_err = np.sqrt(final_mse) / max(y_range, 1e-8) * 100
            print(f"\n  ✓ MLP_{name} — Final Val MSE: {final_mse:.6f}, "
                  f"Relative Error: {rel_err:.2f}%\n")

    # ── Plot grokking curves ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, (name, hist) in enumerate(all_histories.items()):
        ax = axes[idx]
        ax.semilogy(hist["epoch"], hist["train_loss"], label="Train", alpha=0.8)
        ax.semilogy(hist["epoch"], hist["val_loss"], label="Val", alpha=0.8)
        ax.set_title(f"MLP_{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(all_histories) < len(axes):
        for j in range(len(all_histories), len(axes)):
            axes[j].set_visible(False)

    plt.suptitle("Grokking Curves: Train vs Validation Loss", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "grokking_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Grokking curves saved to {plots_dir / 'grokking_curves.png'}")

    # Save histories
    with open(save_dir / "training_histories.json", "w") as f:
        json.dump(all_histories, f, indent=2)

    print("\n✓ All models saved to checkpoints/")
    print("✓ Done!")


if __name__ == "__main__":
    main()
