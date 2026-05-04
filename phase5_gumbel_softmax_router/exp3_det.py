"""
Experiment 3 (deterministic-val variant).

Identical to exp3 in every modeling choice — same router, same complexity prior,
same training schedule. The only differences are:

1. cfg.deterministic_val = True
   Val MSE is evaluated with hard argmax routing (no Gumbel noise). Removes the
   single-draw spikes that made the historical val curve unreliable as a
   point estimate (e.g. epoch 1400 reading 1e-4 between 1e-6 neighbours).

2. get_routing_summary(..., noise=False)
   Final scorecard activations/coefficients are computed deterministically too,
   so commitment / dominant-basis numbers don't depend on a random draw.

3. exp_id = "exp3_det"
   Separate scorecard row and separate HuggingFace upload prefix, so this run
   does not overwrite the historical exp3 artifacts.

Train behavior is byte-identical to exp3 — the noise toggle only affects val
and the final summary.
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .library  import load_library
from .routers  import TopKRouter, build_complexity_prior
from .training import TrainConfig, train_router
from .metrics  import compute_scorecard
from .reporting import print_summary, plot_run, upload_run


def run(load_grokked_mlp, hf_repo_id, device="cpu",
        k=2, alpha=1.0, upload=True):
    """
    Run Experiment 3 with deterministic val evaluation.

    Returns:
        (router, history, summary, scorecard)
    """
    # ── Data ────────────────────────────────────────────────────────────────
    data_path = hf_hub_download(repo_id=hf_repo_id, filename="phase1/data/oscillator_data.npz")
    data = np.load(data_path)
    X, dXdt = data["X"], data["dXdt"]
    truth = {"k": float(data["k"]), "c": float(data["c"])}
    print(f"  True params: k={truth['k']}, c={truth['c']}")
    print(f"  Data shape: X={X.shape}, dXdt={dXdt.shape}\n")

    idx = np.random.default_rng(42).permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train,    X_val    = X[idx[:n_train]],    X[idx[n_train:]]
    dXdt_train, dXdt_val = dXdt[idx[:n_train]], dXdt[idx[n_train:]]

    # ── Router ──────────────────────────────────────────────────────────────
    mlp_modules = load_library(hf_repo_id, load_grokked_mlp)
    mlp_names = [m[0] for m in mlp_modules]
    prior = build_complexity_prior(mlp_names, alpha=alpha)

    router = TopKRouter(state_dim=2, mlp_modules=mlp_modules, k=k, complexity_prior=prior)
    n_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in router.parameters() if not p.requires_grad)
    print(f"  Library: {router.n_mlps} MLPs — {router.mlp_names}")
    print(f"  Complexity prior (alpha={alpha}): {router.complexity_prior.tolist()}")
    print(f"  Trainable params: {n_trainable} | Frozen MLP params: {n_frozen}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    cfg = TrainConfig(
        epochs=4000, lr=3e-3,
        tau_start=5.0, tau_end=0.05, tau_anneal_epochs=2500,
        entropy_weight_max=0.05,
        log_every=200,
        deterministic_val=True,
    )
    print("=" * 60)
    print(f"  TRAINING (Exp 3 DET: Top-{k} Router + Complexity Prior, alpha={alpha})")
    print("=" * 60)
    history = train_router(router, X_train, dXdt_train, X_val, dXdt_val, cfg, device=device)

    # ── Results ──────────────────────────────────────────────────────────────
    router = router.cpu()
    X_t = torch.from_numpy(X.astype(np.float32))
    summary = router.get_routing_summary(X_t, temperature=0.01, noise=False)
    scorecard = compute_scorecard(router, history, X_val, dXdt_val, truth, exp_id="exp3_det")

    print_summary(router, history, summary, truth)
    plot_run(router, history, summary, exp_id="Exp 3 (det val)")

    if upload:
        upload_run(router, history, summary, exp_id="exp3_det", hf_repo_id=hf_repo_id)

    return router, history, summary, scorecard
