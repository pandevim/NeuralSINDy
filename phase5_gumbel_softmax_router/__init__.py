"""
Phase 5: Gumbel-Softmax Neural Router for MLP Library Selection.

Public API
----------
    from phase5_gumbel_softmax_router import exp1, exp2, exp3

    router, history, summary, scorecard = exp1.run(load_grokked_mlp, HF_REPO_ID, device)
    router, history, summary, scorecard = exp2.run(load_grokked_mlp, HF_REPO_ID, device)
    router, history, summary, scorecard = exp3.run(load_grokked_mlp, HF_REPO_ID, device)

Library ablation example
--------------------------
    from phase5_gumbel_softmax_router.library import DEFAULT_SPEC, load_library, drop
    from phase5_gumbel_softmax_router.routers import StateIndepRouter
    from phase5_gumbel_softmax_router.training import TrainConfig, train_router

    mlp_modules = load_library(HF_REPO_ID, load_grokked_mlp, keep_filter=drop(["sin", "cos"]))
    router = StateIndepRouter(state_dim=2, mlp_modules=mlp_modules)
    history = train_router(router, X_train, dXdt_train, X_val, dXdt_val, TrainConfig(), device)
"""

from .library  import DEFAULT_SPEC, load_library, entry_name, drop, keep, apply_filter
from .routers  import (
    gumbel_softmax,
    RouterBase,
    StateDepRouter,
    StateIndepRouter,
    TopKRouter,
    build_complexity_prior,
    spec_from_summary,
)
from .training import TrainConfig, train_router
from .metrics  import compute_scorecard, equation_string
from .reporting import print_summary, plot_run, upload_run

__all__ = [
    # library
    "DEFAULT_SPEC", "load_library", "entry_name", "drop", "keep", "apply_filter",
    # routers
    "gumbel_softmax", "RouterBase",
    "StateDepRouter", "StateIndepRouter", "TopKRouter",
    "build_complexity_prior", "spec_from_summary",
    # training
    "TrainConfig", "train_router",
    # metrics
    "compute_scorecard", "equation_string",
    # reporting
    "print_summary", "plot_run", "upload_run",
]
