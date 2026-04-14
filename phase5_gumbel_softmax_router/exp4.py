"""
Experiment 4: Top-k router with conditional slot entropy.

What Exp 3 got wrong (empirically confirmed)
--------------------------------------------
Exp 3's logit_entropy() penalised the *base* distribution — the logits before
any masking. Minimising that entropy forced the base distribution one-hot on the
dominant term (identity(x) / identity(v)). Once slot 1 committed, the mask
removed that term and left slot 2 facing a nearly-uniform conditional
distribution: the entropy penalty had no gradient signal to differentiate the
remaining candidates. Slot 2 spread across the library and never committed to
identity(v) for dv/dt — the damping term was recovered at only ~-0.028 instead
of the true -0.1, and val MSE plateaued at ~3e-4.

What changes in Experiment 4
-----------------------------
1. Conditional slot entropy (bug fix in TopKRouter.logit_entropy).
   Entropy is now computed slot-by-slot with the mask advanced after each slot,
   exactly mirroring the forward-pass masking logic. Every slot's conditional
   distribution is penalised to commit, not just the global base distribution.
   This gives slot 2 a direct gradient toward concentrating on whichever term
   best explains the residual.

2. Stronger complexity prior (alpha=1.5 vs 1.0).
   In Exp 3's slot-2 spread, sin(v) edged out identity(v) 32.5% vs 28.5%.
   alpha=1.5 gives identity a larger logit advantage (~e^1.5 ≈ 4.5×) that
   more robustly breaks the sin ≈ identity collinearity for small |z|.

3. entropy_weight_max left at 0.05.
   The fixed logit_entropy now sums over k×state_dim=4 conditional entropies
   instead of 2 base entropies, so the effective penalty is ~2× stronger —
   no explicit retuning needed.

Expected outcome
----------------
  dx/dt → identity(v)  at ≈ +1.0   (slot 1 only; slot 2 near-zero)
  dv/dt → identity(x)  at ≈ -1.0   (slot 1)
         + identity(v)  at ≈ -0.1   (slot 2)
  Val MSE → low 1e-5 or better (Exp 3: ~3e-4)
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
        k=2, alpha=1.5, upload=True):
    """
    Run Experiment 4 end-to-end.

    Args:
        load_grokked_mlp: callable(path) -> (model, input_dim)
        hf_repo_id:       HuggingFace repo ID string
        device:           "cpu" or "cuda"
        k:                number of basis functions selected per derivative
        alpha:            complexity prior strength (default 1.5, up from Exp 3's 1.0)
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
    mlp_names   = [m[0] for m in mlp_modules]
    prior       = build_complexity_prior(mlp_names, alpha=alpha)

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
        entropy_weight_max=0.05,   # now covers k×state_dim=4 conditional entropies
        log_every=200,
    )
    print("=" * 60)
    print(f"  TRAINING (Exp 4: Top-{k} Router + Conditional Slot Entropy, alpha={alpha})")
    print("=" * 60)
    history = train_router(router, X_train, dXdt_train, X_val, dXdt_val, cfg, device=device)

    # ── Results ──────────────────────────────────────────────────────────────
    router = router.cpu()
    X_t    = torch.from_numpy(X.astype(np.float32))
    summary   = router.get_routing_summary(X_t, temperature=0.01)
    scorecard = compute_scorecard(router, history, X_val, dXdt_val, truth, exp_id="exp4")

    print_summary(router, history, summary, truth)
    plot_run(router, history, summary, exp_id="Exp 4")

    if upload:
        upload_run(router, history, summary, exp_id="exp4", hf_repo_id=hf_repo_id)

    return router, history, summary, scorecard
