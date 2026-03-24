"""
Run the Gumbel-Softmax Router experiment.

This is the end-to-end differentiable variant of Neural SINDy.
Compare results with run_experiment.py (STLSQ baseline).

Pipeline:
1. Load grokked MLPs (frozen)
2. Load ODE data
3. Train a router network with Gumbel-Softmax to learn which MLP
   explains each component of the dynamics
4. Anneal temperature: soft selection → hard discrete selection
5. Report which MLPs were selected and their learned coefficients
6. Compare against STLSQ baseline results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from neural_router import build_router_from_checkpoints, train_router


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    data_dir = Path("data")

    # ── Load ODE data ──
    print("Loading oscillator data...")
    data = np.load(data_dir / "oscillator_data.npz")
    t = data["t"]
    X = data["X"]
    dXdt = data["dXdt"]
    k_true = float(data["k"])
    c_true = float(data["c"])
    print(f"  True params: k={k_true}, c={c_true}")
    print(f"  Data shape: X={X.shape}, dXdt={dXdt.shape}\n")

    # Train/val split (80/20)
    n = len(t)
    n_train = int(0.8 * n)
    idx = np.random.default_rng(42).permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, X_val = X[train_idx], X[val_idx]
    dXdt_train, dXdt_val = dXdt[train_idx], dXdt[val_idx]

    # ── Build router ──
    print("Building Gumbel-Softmax router from grokked MLPs...")
    router = build_router_from_checkpoints("checkpoints", state_dim=2)

    print(f"  Library size: {router.n_mlps} MLPs")
    print(f"  MLP names: {router.mlp_names}")
    print(f"  Router params: {sum(p.numel() for p in router.parameters() if p.requires_grad)}")
    print(f"  Frozen MLP params: {sum(p.numel() for p in router.parameters() if not p.requires_grad)}\n")

    # ── Train ──
    print("=" * 60)
    print("  TRAINING GUMBEL-SOFTMAX ROUTER")
    print("=" * 60)

    history = train_router(
        router, X_train, dXdt_train, X_val, dXdt_val,
        epochs=3000,
        lr=3e-3,
        tau_start=5.0,
        tau_end=0.05,
        tau_anneal_epochs=2000,
        log_every=200,
        device=device,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  ROUTING RESULTS")
    print("=" * 60)

    # Move to CPU for analysis
    router = router.cpu()
    X_tensor = torch.from_numpy(X.astype(np.float32))

    summary = router.get_routing_summary(X_tensor, temperature=0.01)

    for deriv_name, selected in summary.items():
        print(f"\n  {deriv_name}:")
        for s in selected:
            print(f"    {s['name']:>15s}: "
                  f"selected {s['activation']*100:.1f}% of the time, "
                  f"coefficient = {s['coefficient']:+.4f}")

    # Print as equations
    print("\n" + "=" * 60)
    print("  DISCOVERED EQUATIONS (Gumbel-Softmax)")
    print("=" * 60)

    state_labels = ["ẋ = dx/dt", "v̇ = dv/dt"]
    for d, label in enumerate(state_labels):
        deriv_key = ["dx/dt", "dv/dt"][d]
        selected = summary[deriv_key]
        terms = []
        for s in selected:
            if abs(s["coefficient"]) > 0.01:
                terms.append(f"{s['coefficient']:+.4f} · {s['name']}")
        eq_str = " ".join(terms) if terms else "0"
        print(f"\n  {label} = {eq_str}")

    print(f"\n  TRUE EQUATIONS:")
    print(f"  ẋ = +1.0 · v")
    print(f"  v̇ = -{k_true:.1f} · x  -{c_true:.1f} · v")

    # ── Plot training curves ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    axes[0].semilogy(history["epoch"], history["train_loss"], "b-", alpha=0.8, label="Train")
    axes[0].semilogy(history["epoch"], history["val_loss"], "r-", alpha=0.8, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log)")
    axes[0].set_title("Router Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Temperature schedule
    axes[1].plot(history["epoch"], history["temperature"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Temperature τ")
    axes[1].set_title("Temperature Annealing", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Final routing bar chart
    ax = axes[2]
    n_mlps = router.n_mlps
    x_pos = np.arange(n_mlps)
    width = 0.35

    for d, (deriv_key, color, label) in enumerate(zip(
        ["dx/dt", "dv/dt"], ["#4CAF50", "#F44336"], ["dx/dt", "dv/dt"]
    )):
        activations = np.zeros(n_mlps)
        for s in summary[deriv_key]:
            idx = router.mlp_names.index(s["name"])
            activations[idx] = s["activation"] * s["coefficient"]
        ax.bar(x_pos + d * width, activations, width, color=color,
               edgecolor="black", linewidth=0.5, label=label, alpha=0.8)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(router.mlp_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Activation × Coefficient")
    ax.set_title("Router Selections", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Gumbel-Softmax Router — Training & Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "router_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Router results saved to {plots_dir / 'router_results.png'}")

    # ── Compare with STLSQ if available ──
    stlsq_path = data_dir / "discovery_results.npz"
    if stlsq_path.exists():
        print("\n" + "=" * 60)
        print("  COMPARISON: STLSQ vs GUMBEL-SOFTMAX")
        print("=" * 60)
        stlsq_results = np.load(stlsq_path, allow_pickle=True)
        stlsq_xi = stlsq_results["xi"]
        stlsq_names = list(stlsq_results["term_names"])

        print("\n  STLSQ discovered:")
        for col, label in enumerate(state_labels):
            active = np.abs(stlsq_xi[:, col]) > 1e-10
            terms = [f"{stlsq_xi[i, col]:+.4f}·{stlsq_names[i]}"
                     for i in range(len(stlsq_names)) if active[i]]
            print(f"    {label} = {' '.join(terms) if terms else '0'}")

        print("\n  Gumbel-Softmax discovered:")
        for d, label in enumerate(state_labels):
            deriv_key = ["dx/dt", "dv/dt"][d]
            terms = [f"{s['coefficient']:+.4f}·{s['name']}"
                     for s in summary[deriv_key] if abs(s["coefficient"]) > 0.01]
            print(f"    {label} = {' '.join(terms) if terms else '0'}")

        print(f"\n  TRUE: ẋ = +1.0·v, v̇ = -{k_true:.1f}·x -{c_true:.1f}·v")

    print("\n✓ Gumbel-Softmax experiment complete!")


if __name__ == "__main__":
    main()
