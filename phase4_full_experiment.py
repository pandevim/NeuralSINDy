from huggingface_hub import hf_hub_download

def simulate_discovered_system(xi, term_names, library, x0, t_span, dt):
    """
    Simulate the discovered system forward in time.

    The discovered dynamics are: Ẋ = Θ(X) · ξ
    We numerically integrate this ODE to produce trajectories.
    """
    def rhs(t, state):
        X = state.reshape(1, -1)  # (1, state_dim)
        Theta = library.build_theta(X)  # (1, n_terms)
        dXdt = (Theta @ xi).flatten()  # (state_dim,)
        return dXdt

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method="RK45", max_step=dt)
    return sol.t, sol.y.T

# ── Load data ──
print("Loading oscillator data...")
path = hf_hub_download(repo_id=HF_REPO_ID, filename="phase1/data/oscillator_data.npz")
data = np.load(path)
t = data["t"]
X = data["X"]
dXdt = data["dXdt"]
k_true = float(data["k"])
c_true = float(data["c"])
print(f"  Loaded {len(t)} time points, true params: k={k_true}, c={c_true}\n")

# ── Build library ──
print("Building Neural SINDy library from grokked MLPs...")
library = build_default_library(HF_REPO_ID)

# ── Discover equations ──
print("\n" + "=" * 60)
print("  NEURAL SINDy DISCOVERY")
print("=" * 60)

# Try a range of thresholds and pick the sparsest good one
best_xi = None
best_threshold = None
best_error = float("inf")

for threshold in [0.01, 0.025, 0.05, 0.1, 0.2]:
    xi, term_names = discover_equations(
        X, dXdt, library,
        threshold=threshold,
        alpha=0.01,
    )

    # Evaluate reconstruction error
    Theta = library.build_theta(X)
    dXdt_pred = Theta @ xi
    mse = np.mean((dXdt - dXdt_pred) ** 2)
    n_active = np.sum(np.abs(xi) > 1e-10)

    print(f"  Threshold={threshold:.3f}: MSE={mse:.6f}, "
          f"Active terms={n_active}/{xi.size}")

    # Pick the sparsest model with acceptable error
    if mse < 0.01 and (best_xi is None or n_active <= np.sum(np.abs(best_xi) > 1e-10)):
        best_xi = xi.copy()
        best_threshold = threshold
        best_error = mse

if best_xi is None:
    # Fall back to lowest error
    best_xi = xi
    best_threshold = threshold
    best_error = mse
    print("\n  ⚠ Could not find a sparse model with low error, using last result")

xi = best_xi
print(f"\n  ✓ Best threshold: {best_threshold}, MSE: {best_error:.6f}")

# ── Print final discovered equations ──
print("\n" + "=" * 60)
print("  DISCOVERED EQUATIONS")
print("=" * 60)

state_labels = ["ẋ = dx/dt", "v̇ = dv/dt"]
for col in range(dXdt.shape[1]):
    active = np.abs(xi[:, col]) > 1e-10
    terms = []
    for i in range(len(term_names)):
        if active[i]:
            coeff = xi[i, col]
            terms.append(f"{coeff:+.4f} · {term_names[i]}")

    eq_str = " ".join(terms) if terms else "0"
    print(f"\n  {state_labels[col]} = {eq_str}")

print(f"\n  TRUE EQUATIONS:")
print(f"  ẋ = +1.0 · v")
print(f"  v̇ = -{k_true:.1f} · x  -{c_true:.1f} · v")

# ── Save results ──
buf = io.BytesIO()
np.savez(buf, xi=xi, term_names=np.array(term_names), best_threshold=best_threshold, mse=best_error)
api.upload_file(path_or_fileobj=io.BytesIO(buf.getvalue()),
                path_in_repo="phase1/data/discovery_results.npz",
                repo_id=HF_REPO_ID, commit_message="Add discovery results")

# ── Simulate discovered system ──
print("\n\nSimulating discovered system...")
try:
    t_sim, X_sim = simulate_discovered_system(
        xi, term_names, library,
        x0=X[0],
        t_span=(t[0], t[-1]),
        dt=0.05,
    )

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(t, X[:, 0], "b-", alpha=0.8, label="True x(t)")
    axes[0].plot(t_sim, X_sim[:, 0], "r--", alpha=0.8, label="Discovered x(t)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Position x")
    axes[0].set_title("Position: True vs Discovered")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, X[:, 1], "b-", alpha=0.8, label="True v(t)")
    axes[1].plot(t_sim, X_sim[:, 1], "r--", alpha=0.8, label="Discovered v(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Velocity v")
    axes[1].set_title("Velocity: True vs Discovered")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Neural SINDy: True vs Discovered Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    api.upload_file(path_or_fileobj=io.BytesIO(buf.getvalue()),
                    path_in_repo="phase1/plots/discovery_comparison.png",
                    repo_id=HF_REPO_ID, commit_message="Add discovery comparison plot")
    print("  ✓ Comparison plot uploaded to HuggingFace")
except Exception as e:
    print(f"  ⚠ Simulation failed (this is OK for PoC): {e}")

# ── Coefficient bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x_pos = np.arange(len(term_names))

for col, (ax, label) in enumerate(zip(axes, state_labels)):
    colors = ["#2196F3" if abs(v) > 1e-10 else "#BDBDBD" for v in xi[:, col]]
    ax.bar(x_pos, xi[:, col], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(term_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Coefficient ξ")
    ax.set_title(f"Sparse Coefficients for {label}")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Neural SINDy: Selected Library Terms", fontsize=14, fontweight="bold")
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close()
api.upload_file(path_or_fileobj=io.BytesIO(buf.getvalue()),
                path_in_repo="phase1/plots/coefficients.png",
                repo_id=HF_REPO_ID, commit_message="Add coefficients plot")
print("  ✓ Coefficient bar chart uploaded to HuggingFace")

print("\n✓ Experiment complete!")