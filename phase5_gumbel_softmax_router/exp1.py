"""
Experiment 1: State-dependent Gumbel-Softmax router.

Architecture
------------
    State [x, v]  →  Router MLP  →  logits  →  Gumbel-Softmax gate
    gate (one-hot, STE)  ×  coefficients  ×  MLP_i(state)  →  dXdt_pred

The router is a small neural net that maps the current state to routing
logits. Because logits vary per sample, the same basis function need not
explain the dynamics everywhere — this violates the SINDy assumption and
is the main reason Exp 1 fails to commit cleanly.

Key result: max activation ~44% (cos/identity confusion, no commitment).
Leads to Exp 2 (state-independent routing).
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .library  import load_library
from .routers  import StateDepRouter
from .training import TrainConfig, train_router
from .metrics  import compute_scorecard
from .reporting import print_summary, plot_run, upload_run


def run(load_grokked_mlp, hf_repo_id, device="cpu", upload=True):
    """
    Run Experiment 1 end-to-end.

    Args:
        load_grokked_mlp: callable(path) -> (model, input_dim)
                          Use the project's load_grokked_mlp from phase3.
        hf_repo_id:       HuggingFace repo ID string
        device:           "cpu" or "cuda"
        upload:           upload checkpoint / results / plot to HuggingFace

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
    router = StateDepRouter(state_dim=2, mlp_modules=mlp_modules, router_hidden=64)
    n_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in router.parameters() if not p.requires_grad)
    print(f"  Library: {router.n_mlps} MLPs — {router.mlp_names}")
    print(f"  Trainable params: {n_trainable} | Frozen MLP params: {n_frozen}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    # entropy_weight_max=0: Exp 1 used a gate-based exploration bonus instead
    # of the commitment penalty used in Exp 2/3. The bonus is omitted here
    # for simplicity; the state-dep router explores naturally via its MLP.
    cfg = TrainConfig(
        epochs=3000, lr=3e-3,
        tau_start=5.0, tau_end=0.05, tau_anneal_epochs=2000,
        entropy_weight_max=0.0,
        log_every=200,
    )
    print("=" * 60)
    print("  TRAINING (Exp 1: State-Dependent Router)")
    print("=" * 60)
    history = train_router(router, X_train, dXdt_train, X_val, dXdt_val, cfg, device=device)

    # ── Results ──────────────────────────────────────────────────────────────
    router = router.cpu()
    X_t = torch.from_numpy(X.astype(np.float32))
    summary = router.get_routing_summary(X_t, temperature=0.01)
    scorecard = compute_scorecard(router, history, X_val, dXdt_val, truth, exp_id="exp1")

    print_summary(router, history, summary, truth)
    plot_run(router, history, summary, exp_id="Exp 1")

    if upload:
        upload_run(router, history, summary, exp_id="exp1", hf_repo_id=hf_repo_id)

    return router, history, summary, scorecard
