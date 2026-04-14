"""
Visualization: Generate publication-quality plots for the experiment.

Run this after all other phases to produce a comprehensive visual summary.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json

from neural_sindy import load_grokked_mlp


def plot_mlp_approximations(ckpt_dir="checkpoints", plots_dir="plots"):
    """Plot each grokked MLP's output vs the ground truth function."""
    ckpt_dir = Path(ckpt_dir)
    plots_dir = Path(plots_dir)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    ground_truth = {
        "identity": (lambda x: x, "f(x) = x", (-5, 5)),
        "cos":      (lambda x: np.cos(x), "f(x) = cos(x)", (-2*np.pi, 2*np.pi)),
        "sin":      (lambda x: np.sin(x), "f(x) = sin(x)", (-2*np.pi, 2*np.pi)),
    }

    idx = 0

    # Unary functions
    for name, (func, label, x_range) in ground_truth.items():
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            continue

        model, _ = load_grokked_mlp(path)
        x = np.linspace(x_range[0], x_range[1], 500).reshape(-1, 1).astype(np.float32)
        y_true = func(x).flatten()

        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(x)).numpy().flatten()

        ax = axes[idx]
        ax.plot(x.flatten(), y_true, "b-", linewidth=2, label=f"True: {label}", alpha=0.8)
        ax.plot(x.flatten(), y_pred, "r--", linewidth=2, label=f"MLP_{name}", alpha=0.8)

        # Error band
        error = np.abs(y_true - y_pred)
        ax.fill_between(x.flatten(), y_true - error, y_true + error, alpha=0.15, color="red")

        mse = np.mean((y_true - y_pred) ** 2)
        ax.set_title(f"MLP_{name} (MSE: {mse:.2e})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        idx += 1

    # Binary functions — show as heatmap difference
    binary_fns = {
        "add": (lambda x, y: x + y, "f(x,y) = x + y"),
        "mul": (lambda x, y: x * y, "f(x,y) = x · y"),
    }

    for name, (func, label) in binary_fns.items():
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            continue

        model, _ = load_grokked_mlp(path)

        x1 = np.linspace(-5, 5, 50)
        x2 = np.linspace(-5, 5, 50)
        X1, X2 = np.meshgrid(x1, x2)
        x_in = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)

        y_true = func(X1.ravel(), X2.ravel())
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.from_numpy(x_in)).numpy().flatten()

        error = np.abs(y_true - y_pred).reshape(50, 50)
        mse = np.mean((y_true - y_pred) ** 2)

        ax = axes[idx]
        im = ax.imshow(error, extent=[-5, 5, -5, 5], origin="lower",
                       cmap="hot_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Absolute Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"MLP_{name} Error Map (MSE: {mse:.2e})", fontsize=12, fontweight="bold")
        idx += 1

    # Hide unused
    for j in range(idx, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Grokked MLP Approximations vs Ground Truth",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "mlp_approximations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ MLP approximations saved to {plots_dir / 'mlp_approximations.png'}")


def plot_full_summary(data_dir="data", plots_dir="plots"):
    """Create a combined summary figure."""
    data_dir = Path(data_dir)
    plots_dir = Path(plots_dir)

    # Load discovery results
    results_path = data_dir / "discovery_results.npz"
    if not results_path.exists():
        print("  ⚠ No discovery results found, run run_experiment.py first")
        return

    results = np.load(results_path, allow_pickle=True)
    xi = results["xi"]
    term_names = list(results["term_names"])

    # Load oscillator data
    osc_data = np.load(data_dir / "oscillator_data.npz")
    t = osc_data["t"]
    X = osc_data["X"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Phase portrait
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X[:, 0], X[:, 1], "b-", alpha=0.7, linewidth=0.8)
    ax1.plot(X[0, 0], X[0, 1], "go", markersize=10, label="Start", zorder=5)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Velocity v")
    ax1.set_title("Phase Portrait", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Position time series
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, X[:, 0], "b-", alpha=0.8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("x(t)")
    ax2.set_title("Position vs Time", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Velocity time series
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, X[:, 1], "r-", alpha=0.8)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("v(t)")
    ax3.set_title("Velocity vs Time", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # 4. Coefficients for dx/dt
    ax4 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(term_names))
    colors = ["#4CAF50" if abs(v) > 1e-10 else "#E0E0E0" for v in xi[:, 0]]
    ax4.bar(x_pos, xi[:, 0], color=colors, edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(term_names, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("ξ")
    ax4.set_title("ẋ = dx/dt coefficients", fontweight="bold")
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Coefficients for dv/dt
    ax5 = fig.add_subplot(gs[1, 1])
    colors = ["#F44336" if abs(v) > 1e-10 else "#E0E0E0" for v in xi[:, 1]]
    ax5.bar(x_pos, xi[:, 1], color=colors, edgecolor="black", linewidth=0.5)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(term_names, rotation=45, ha="right", fontsize=8)
    ax5.set_ylabel("ξ")
    ax5.set_title("v̇ = dv/dt coefficients", fontweight="bold")
    ax5.axhline(y=0, color="black", linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Discovered equations text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    eq_text = "DISCOVERED EQUATIONS\n" + "=" * 30 + "\n\n"

    state_labels = ["ẋ = dx/dt", "v̇ = dv/dt"]
    for col in range(2):
        active = np.abs(xi[:, col]) > 1e-10
        terms = []
        for i in range(len(term_names)):
            if active[i]:
                coeff = xi[i, col]
                terms.append(f"{coeff:+.3f}·{term_names[i]}")
        eq_str = " ".join(terms) if terms else "0"
        eq_text += f"{state_labels[col]} = {eq_str}\n\n"

    eq_text += "\nTRUE EQUATIONS\n" + "=" * 30 + "\n\n"
    eq_text += "ẋ = +1.0·v\n\n"
    eq_text += f"v̇ = -{float(osc_data['k']):.1f}·x - {float(osc_data['c']):.1f}·v\n"

    ax6.text(0.05, 0.95, eq_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.8))

    plt.suptitle("Neural SINDy with Grokked MLPs — Full Summary",
                 fontsize=16, fontweight="bold")
    plt.savefig(plots_dir / "full_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Full summary saved to {plots_dir / 'full_summary.png'}")


def main():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_mlp_approximations()
    plot_full_summary()

    print("\n✓ All visualizations saved to plots/")
    print("✓ Done!")


if __name__ == "__main__":
    main()
