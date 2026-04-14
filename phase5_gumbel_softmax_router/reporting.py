"""
Reporting utilities: console output, plots, and HuggingFace uploads.

Each experiment driver calls the same three functions at the end of a run:
    print_summary(router, history, summary, truth)
    plot_run(router, history, summary, exp_id)
    upload_run(router, history, summary, exp_id, hf_repo_id)   # optional

Artifacts are stored under phase5/{exp_id}/ on HuggingFace so every experiment
gets its own namespace and nothing gets clobbered:
    phase5/exp1/router.pt
    phase5/exp1/results.npz
    phase5/exp1/plot.png
    phase5/exp2/router.pt
    ...
"""

from .metrics import equation_string

def print_summary(router, history, summary, truth):
    """Print routing results, discovered equations, and true equations."""
    print("\n" + "=" * 60)
    print("  ROUTING RESULTS")
    print("=" * 60)
    for deriv, terms in summary.items():
        print(f"\n  {deriv}:")
        for t in terms:
            print(
                f"    {t['name']:>15s}: "
                f"selected {t['activation'] * 100:.1f}% of the time, "
                f"coefficient = {t['coefficient']:+.4f}"
            )

    print("\n" + "=" * 60)
    print("  DISCOVERED EQUATIONS")
    print("=" * 60)
    _LABELS = {"dx/dt": "ẋ = dx/dt", "dv/dt": "v̇ = dv/dt"}
    for deriv, terms in summary.items():
        label = _LABELS.get(deriv, deriv)
        print(f"\n  {label} = {equation_string(terms)}")

    print(f"\n  TRUE EQUATIONS:")
    print(f"  ẋ = +1.0 · identity(v)")
    print(f"  v̇ = -{truth['k']:.1f} · identity(x)  -{truth['c']:.1f} · identity(v)")


def plot_run(router, history, summary, exp_id, show=True, return_fig=False):
    """
    Standard 3-panel diagnostic plot:
        left   — train/val loss curves (log scale)
        centre — temperature annealing schedule
        right  — routing bar chart (activation × coefficient per MLP)

    Args:
        show:       call plt.show() (set False in upload path)
        return_fig: return the Figure object instead of closing it
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    axes[0].semilogy(history["epoch"], history["train_loss"], "b-", alpha=0.8, label="Train")
    axes[0].semilogy(history["epoch"], history["val_loss"],   "r-", alpha=0.8, label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log)")
    axes[0].set_title(f"{exp_id}: Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Temperature schedule
    axes[1].plot(history["epoch"], history["temperature"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Temperature τ")
    axes[1].set_title("Temperature Annealing", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Routing bar chart
    ax = axes[2]
    n_mlps = router.n_mlps
    x_pos = np.arange(n_mlps)
    width = 0.35
    for d, (deriv_key, color) in enumerate(
        zip(["dx/dt", "dv/dt"], ["#4CAF50", "#F44336"])
    ):
        vals = np.zeros(n_mlps)
        for s in summary.get(deriv_key, []):
            i = router.mlp_names.index(s["name"])
            vals[i] = s["activation"] * s["coefficient"]
        ax.bar(
            x_pos + d * width, vals, width,
            color=color, edgecolor="black", linewidth=0.5,
            label=deriv_key, alpha=0.8,
        )
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(router.mlp_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Activation × Coefficient")
    ax.set_title("Router Selections", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Phase 5 · {exp_id} — Gumbel-Softmax Router",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)


def upload_run(router, history, summary, exp_id, hf_repo_id):
    """
    Upload checkpoint, results npz, and plot png to HuggingFace.
    All artifacts land under phase5/{exp_id}/ in the repo.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    prefix = f"phase5/{exp_id}"

    # ── Checkpoint ──
    buf = io.BytesIO()
    ckpt = {
        "router_state_dict": router.state_dict(),
        "mlp_names": router.mlp_names,
        "state_dim": router.state_dim,
        "router_class": type(router).__name__,
    }
    if hasattr(router, "k"):
        ckpt["k"] = router.k
    if hasattr(router, "complexity_prior"):
        ckpt["complexity_prior"] = router.complexity_prior.cpu().numpy()
    torch.save(ckpt, buf)
    api.upload_file(
        path_or_fileobj=io.BytesIO(buf.getvalue()),
        path_in_repo=f"{prefix}/router.pt",
        repo_id=hf_repo_id,
        commit_message=f"Phase 5 {exp_id}: router checkpoint",
    )

    # ── Results npz ──
    buf = io.BytesIO()
    np.savez(buf,
             mlp_names=np.array(router.mlp_names),
             summary=np.array(summary, dtype=object),
             history=np.array(history, dtype=object))
    api.upload_file(
        path_or_fileobj=io.BytesIO(buf.getvalue()),
        path_in_repo=f"{prefix}/results.npz",
        repo_id=hf_repo_id,
        commit_message=f"Phase 5 {exp_id}: results",
    )

    # ── Plot ──
    fig = plot_run(router, history, summary, exp_id, show=False, return_fig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    api.upload_file(
        path_or_fileobj=io.BytesIO(buf.getvalue()),
        path_in_repo=f"{prefix}/plot.png",
        repo_id=hf_repo_id,
        commit_message=f"Phase 5 {exp_id}: plot",
    )

    print(f"  Uploaded checkpoint, results, and plot to {hf_repo_id}/{prefix}/")
