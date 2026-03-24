"""
Phase 3: Generate synthetic ODE data for testing.

We use a damped harmonic oscillator:
    ẍ = -k·x - c·ẋ

Written as a first-order system:
    dx/dt = v
    dv/dt = -k·x - c·v

This system involves linear combinations of state variables,
which is exactly what our MLP library should discover.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path


def generate_damped_oscillator(
    k=1.0, c=0.1,
    x0=1.0, v0=0.0,
    t_span=(0, 30), dt=0.05,
    noise_std=0.001,
    seed=42,
):
    """
    Generate time-series data from a damped harmonic oscillator.

    Returns:
        t: time points (N,)
        X: state matrix [x, v] of shape (N, 2)
        dXdt: derivatives [dx/dt, dv/dt] of shape (N, 2)
        params: dict with k, c
    """
    rng = np.random.default_rng(seed)

    def rhs(t, state):
        x, v = state
        dxdt = v
        dvdt = -k * x - c * v
        return [dxdt, dvdt]

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(rhs, t_span, [x0, v0], t_eval=t_eval, method="RK45", max_step=dt/2)

    t = sol.t
    X = sol.y.T  # (N, 2) — columns are [x, v]

    # Compute clean derivatives
    dXdt = np.zeros_like(X)
    dXdt[:, 0] = X[:, 1]                    # dx/dt = v
    dXdt[:, 1] = -k * X[:, 0] - c * X[:, 1]  # dv/dt = -k·x - c·v

    # Add measurement noise to state (NOT to derivatives)
    X_noisy = X + rng.normal(0, noise_std, size=X.shape)

    return t, X_noisy, dXdt, {"k": k, "c": c}


def main():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Generating damped harmonic oscillator data...")
    print("  ẍ = -k·x - c·ẋ  with k=1.0, c=0.1")
    print("  x(0) = 1.0, ẋ(0) = 0.0\n")

    t, X, dXdt, params = generate_damped_oscillator()

    print(f"  Time points:  {len(t)}")
    print(f"  State shape:  {X.shape}")
    print(f"  dX/dt shape:  {dXdt.shape}")
    print(f"  Parameters:   k={params['k']}, c={params['c']}")

    # Save data
    np.savez(
        data_dir / "oscillator_data.npz",
        t=t, X=X, dXdt=dXdt,
        k=params["k"], c=params["c"],
    )
    print(f"\n✓ Data saved to {data_dir / 'oscillator_data.npz'}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Phase portrait
    axes[0, 0].plot(X[:, 0], X[:, 1], "b-", alpha=0.7, linewidth=0.8)
    axes[0, 0].plot(X[0, 0], X[0, 1], "go", markersize=8, label="Start")
    axes[0, 0].plot(X[-1, 0], X[-1, 1], "rs", markersize=8, label="End")
    axes[0, 0].set_xlabel("x (position)")
    axes[0, 0].set_ylabel("v (velocity)")
    axes[0, 0].set_title("Phase Portrait")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Time series - position
    axes[0, 1].plot(t, X[:, 0], "b-", alpha=0.8, label="x(t)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Position x")
    axes[0, 1].set_title("Position vs Time")
    axes[0, 1].grid(True, alpha=0.3)

    # Time series - velocity
    axes[1, 0].plot(t, X[:, 1], "r-", alpha=0.8, label="v(t)")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Velocity v")
    axes[1, 0].set_title("Velocity vs Time")
    axes[1, 0].grid(True, alpha=0.3)

    # Derivatives
    axes[1, 1].plot(t, dXdt[:, 0], "b-", alpha=0.6, label="dx/dt")
    axes[1, 1].plot(t, dXdt[:, 1], "r-", alpha=0.6, label="dv/dt")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Derivative")
    axes[1, 1].set_title("True Derivatives")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Damped Harmonic Oscillator: ẍ = -{params['k']}·x - {params['c']}·ẋ",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "oscillator_data.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to {plots_dir / 'oscillator_data.png'}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
