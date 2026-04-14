"""
Phase 5: Symbolic Distillation — extract human-readable equations from grokked MLPs.

This implements the "Noiseless Oracle" method:
1. Take each winning MLP from the SINDy discovery
2. Feed it a clean, noise-free grid of inputs
3. Run PySR symbolic regression on the (input, output) pairs
4. Because there's zero noise, PySR should instantly identify the function

This is the key step that closes the loop:
    Noisy real data → Neural SINDy selects MLPs → Distill MLPs back to math
"""

import numpy as np
import torch
from pathlib import Path

# PySR import with graceful fallback
try:
    from pysr import PySRRegressor
    HAS_PYSR = True
except ImportError:
    HAS_PYSR = False
    print("⚠ PySR not installed. Install with: pip install pysr")
    print("  (Requires Julia — PySR will auto-install it on first run)")

from phase3_neural_sindy import load_grokked_mlp


def distill_unary_mlp(model, name, x_range=(-6.0, 6.0), n_points=500):
    """
    Distill a unary MLP (1 input → 1 output) into a symbolic expression.
    """
    print(f"\n  Distilling MLP_{name}...")

    # Generate clean synthetic data
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x)).numpy().flatten()

    print(f"    Input range: [{x_range[0]}, {x_range[1]}]")
    print(f"    Output range: [{y.min():.4f}, {y.max():.4f}]")

    if not HAS_PYSR:
        print("    ⚠ Skipping PySR (not installed). Showing input-output statistics only.")
        return None

    # Run PySR
    pysr_model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["cos", "sin", "exp", "log", "abs", "square"],
        populations=20,
        population_size=50,
        maxsize=15,
        parsimony=0.01,  # Prefer simpler expressions
        timeout_in_seconds=120,
        temp_equation_file=True,
        verbosity=0,
    )

    pysr_model.fit(x, y)

    # Get best equation
    best_eq = pysr_model.get_best()
    print(f"    ✓ Best equation: {best_eq['equation']}")
    print(f"      Loss: {best_eq['loss']:.8f}")
    print(f"      Complexity: {best_eq['complexity']}")

    # Verify
    y_pred = pysr_model.predict(x)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"      R²: {r2:.6f}")

    return {
        "name": name,
        "equation": str(best_eq["equation"]),
        "loss": float(best_eq["loss"]),
        "complexity": int(best_eq["complexity"]),
        "r2": float(r2),
    }


def distill_binary_mlp(model, name, x_range=(-5.0, 5.0), n_points_per_dim=50):
    """
    Distill a binary MLP (2 inputs → 1 output) into a symbolic expression.
    """
    print(f"\n  Distilling MLP_{name}...")

    # Generate 2D grid
    x1 = np.linspace(x_range[0], x_range[1], n_points_per_dim)
    x2 = np.linspace(x_range[0], x_range[1], n_points_per_dim)
    X1, X2 = np.meshgrid(x1, x2)
    x_in = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)

    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x_in)).numpy().flatten()

    print(f"    Input range: [{x_range[0]}, {x_range[1]}] × [{x_range[0]}, {x_range[1]}]")
    print(f"    Output range: [{y.min():.4f}, {y.max():.4f}]")

    if not HAS_PYSR:
        print("    ⚠ Skipping PySR (not installed).")
        return None

    pysr_model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["cos", "sin", "exp", "square"],
        populations=20,
        population_size=50,
        maxsize=10,
        parsimony=0.01,
        timeout_in_seconds=120,
        temp_equation_file=True,
        verbosity=0,
    )

    pysr_model.fit(x_in, y)

    best_eq = pysr_model.get_best()
    print(f"    ✓ Best equation: {best_eq['equation']}")
    print(f"      Loss: {best_eq['loss']:.8f}")
    print(f"      Complexity: {best_eq['complexity']}")

    y_pred = pysr_model.predict(x_in)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"      R²: {r2:.6f}")

    return {
        "name": name,
        "equation": str(best_eq["equation"]),
        "loss": float(best_eq["loss"]),
        "complexity": int(best_eq["complexity"]),
        "r2": float(r2),
    }


def main():
    ckpt_dir = Path("checkpoints")

    print("=" * 60)
    print("  SYMBOLIC DISTILLATION")
    print("  Extracting math from grokked MLPs")
    print("=" * 60)

    results = []

    # Distill all unary MLPs
    for name in ["identity", "cos", "sin"]:
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            print(f"\n  ⚠ {path} not found, skipping")
            continue

        model, input_dim = load_grokked_mlp(path)
        result = distill_unary_mlp(model, name)
        if result:
            results.append(result)

    # Distill binary MLPs
    for name in ["add", "mul"]:
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            print(f"\n  ⚠ {path} not found, skipping")
            continue

        model, input_dim = load_grokked_mlp(path)
        result = distill_binary_mlp(model, name)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("  DISTILLATION RESULTS")
        print("=" * 60)
        print(f"\n  {'MLP':<12} {'Recovered Equation':<30} {'R²':<10} {'Loss':<12}")
        print("  " + "-" * 64)
        for r in results:
            print(f"  {r['name']:<12} {r['equation']:<30} {r['r2']:<10.6f} {r['loss']:<12.8f}")

    print("\n✓ Distillation complete!")


if __name__ == "__main__":
    main()
