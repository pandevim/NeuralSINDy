"""
Scorecard metrics for comparing router experiments.

Every experiment returns a scorecard dict with the same keys, so comparison
across runs is just pd.DataFrame([sc1, sc2, sc3]).

Scorecard keys
--------------
    exp_id              str   — experiment label, e.g. "exp1"
    n_params            int   — trainable parameter count
    val_mse             float — MSE on validation set at τ=0.01
    commitment          float — fraction of samples with max gate ≥ 90%
    equations           dict  — {deriv: equation_string}
    correctness         bool  — True if dominant terms match the true ODE
    final_tau           float — last logged temperature value
"""

def compute_scorecard(router, history, X_val, dXdt_val, truth, exp_id=""):
    """
    Compute the standard scorecard for one router run.

    Args:
        router:    trained RouterBase subclass, on CPU
        history:   dict returned by train_router
        X_val:     numpy (N, state_dim) — validation set
        dXdt_val:  numpy (N, state_dim)
        truth:     dict with "k" and "c" (true oscillator params)
        exp_id:    label string for the scorecard row

    Returns:
        dict with scorecard fields
    """
    router.eval()
    X_t  = torch.from_numpy(X_val.astype(np.float32))
    dX_t = torch.from_numpy(dXdt_val.astype(np.float32))

    with torch.no_grad():
        pred, _ = router(X_t, temperature=0.01, hard=True)
        val_mse = F.mse_loss(pred, dX_t).item()

    summary = router.get_routing_summary(X_t, temperature=0.01)

    return {
        "exp_id":      exp_id,
        "n_params":    sum(p.numel() for p in router.parameters() if p.requires_grad),
        "val_mse":     val_mse,
        "commitment":  _commitment_score(router, X_t),
        "equations":   {deriv: equation_string(terms) for deriv, terms in summary.items()},
        "correctness": _check_correctness(summary, truth),
        "final_tau":   history["temperature"][-1] if history["temperature"] else None,
    }


def equation_string(terms, threshold=0.01):
    """
    Format a list of routing summary term dicts as a human-readable equation.

    Example: "+1.0023·identity(v)"
    """
    active = [t for t in terms if abs(t["coefficient"]) > threshold]
    if not active:
        return "0"
    return " ".join(f"{t['coefficient']:+.4f}·{t['name']}" for t in active)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _commitment_score(router, X_t):
    """
    Fraction of samples where the dominant gate has ≥ 90% activation.
    A committed router has score ≈ 1.0; a spread router has score ≈ 0.
    """
    with torch.no_grad():
        _, gates = router(X_t, temperature=0.01, hard=True)

    # Normalise gate shape across router variants.
    # StateDepRouter / StateIndepRouter: (state_dim, batch, n_mlps)
    # TopKRouter:                        (state_dim, k, batch, n_mlps) — use first slot
    if gates.dim() == 4:
        gates = gates[:, 0]  # (state_dim, batch, n_mlps)

    max_act = gates.max(dim=-1).values  # (state_dim, batch)
    return (max_act >= 0.9).float().mean().item()


def _check_correctness(summary, truth):
    """
    True if the dominant discovered terms match the harmonic oscillator ODE.

    True equations:
        dx/dt =  +1.0 · identity(v)
        dv/dt =  -k   · identity(x) − c · identity(v)

    Checks:
        dx/dt: identity(v) present with coefficient > 0.5
        dv/dt: identity(x) present with negative coefficient,
               identity(v) present with negative coefficient
    """
    dx_terms = {t["name"]: t["coefficient"] for t in summary.get("dx/dt", [])}
    dv_terms = {t["name"]: t["coefficient"] for t in summary.get("dv/dt", [])}

    dx_ok = "identity(v)" in dx_terms and dx_terms["identity(v)"] > 0.5
    dv_ok = (
        "identity(x)" in dv_terms and dv_terms["identity(x)"] < -0.1
        and "identity(v)" in dv_terms and dv_terms["identity(v)"] < -0.1
    )
    return dx_ok and dv_ok
