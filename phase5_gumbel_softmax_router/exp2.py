"""
Experiment 2: State-independent Gumbel-Softmax router.

What changed from Exp 1
-----------------------
1. State-independent routing.
   The router is now a single learnable logit vector per derivative,
   broadcast to the full batch. Every sample gets the same global routing
   decision (before Gumbel noise), matching SINDy's assumption that one
   term governs the dynamics everywhere in state space.
   Param count drops from ~9 760 → 16 (2 derivatives × 8 MLPs).

2. Commitment entropy penalty.
   Instead of an exploration bonus (Exp 1), we now penalise spread:
       loss += w(τ) × H(softmax(logits))
   where w(τ) ramps from 0 → entropy_weight_max as τ anneals.
   Late in training, the router is forced to commit to a single MLP.

Key result: clean one-hot commitment (>99%), but selects sin(x)/sin(v)
instead of identity(x)/identity(v) due to approximate collinearity
(sin(z) ≈ z for |z| ≲ 1). Also structurally limited to 1 term per
derivative — the true v̇ = -k·x - c·v needs two terms.
Leads to Exp 3 (top-k routing + complexity prior).
"""

from .library  import load_library
from .routers  import StateIndepRouter
from .training import TrainConfig, train_router
from .metrics  import compute_scorecard
from .reporting import print_summary, plot_run, upload_run


def run(load_grokked_mlp, hf_repo_id, device="cpu", upload=True):
    """
    Run Experiment 2 end-to-end.

    Args:
        load_grokked_mlp: callable(path) -> (model, input_dim)
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
    router = StateIndepRouter(state_dim=2, mlp_modules=mlp_modules)
    n_trainable = sum(p.numel() for p in router.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in router.parameters() if not p.requires_grad)
    print(f"  Library: {router.n_mlps} MLPs — {router.mlp_names}")
    print(f"  Trainable params: {n_trainable} | Frozen MLP params: {n_frozen}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    cfg = TrainConfig(
        epochs=3000, lr=3e-3,
        tau_start=5.0, tau_end=0.05, tau_anneal_epochs=2000,
        entropy_weight_max=0.05,
        log_every=200,
    )
    print("=" * 60)
    print("  TRAINING (Exp 2: State-Independent Router)")
    print("=" * 60)
    history = train_router(router, X_train, dXdt_train, X_val, dXdt_val, cfg, device=device)

    # ── Results ──────────────────────────────────────────────────────────────
    router = router.cpu()
    X_t = torch.from_numpy(X.astype(np.float32))
    summary = router.get_routing_summary(X_t, temperature=0.01)
    scorecard = compute_scorecard(router, history, X_val, dXdt_val, truth, exp_id="exp2")

    print_summary(router, history, summary, truth)
    plot_run(router, history, summary, exp_id="Exp 2")

    if upload:
        upload_run(router, history, summary, exp_id="exp2", hf_repo_id=hf_repo_id)

    return router, history, summary, scorecard
