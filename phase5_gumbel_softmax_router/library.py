"""
Declarative basis-function library for the Gumbel-Softmax router.

The library is a list of entry dicts:
    {"basis": "identity", "arg": "x", "cols": [0]}
    {"basis": "sin",      "arg": "v", "cols": [1]}
    {"basis": "add",      "arg": "x,v", "cols": [0, 1]}

`load_library` turns these specs into (name, model, cols) tuples that every
router constructor accepts. A fresh model instance is loaded per entry, so
sin(x) and sin(v) are independent modules even though they share a checkpoint.

Library filters
---------------
    drop(["sin", "cos"])          → remove all sin/cos entries
    keep(["identity", "add"])     → keep only identity and add entries
    apply_filter(spec, fn)        → apply a filter function to a spec

Usage example (library ablation in a notebook)
-----------------------------------------------
    from phase5_gumbel_softmax_router.library import (
        DEFAULT_SPEC, load_library, apply_filter, drop
    )
    spec = apply_filter(DEFAULT_SPEC, drop(["sin", "cos"]))
    mlp_modules = load_library(HF_REPO_ID, load_grokked_mlp, spec=spec)
"""

# Ordered the same way as the original build_router_from_checkpoints:
# unary functions (identity, cos, sin) × each state variable, then binary functions.
DEFAULT_SPEC = [
    {"basis": "identity", "arg": "x",   "cols": [0]},
    {"basis": "identity", "arg": "v",   "cols": [1]},
    {"basis": "cos",      "arg": "x",   "cols": [0]},
    {"basis": "cos",      "arg": "v",   "cols": [1]},
    {"basis": "sin",      "arg": "x",   "cols": [0]},
    {"basis": "sin",      "arg": "v",   "cols": [1]},
    {"basis": "add",      "arg": "x,v", "cols": [0, 1]},
    {"basis": "mul",      "arg": "x,v", "cols": [0, 1]},
]


def entry_name(entry):
    """Return the display name for a library entry, e.g. 'sin(v)'."""
    return f"{entry['basis']}({entry['arg']})"


def load_library(repo_id, load_mlp_fn, spec=None, keep_filter=None):
    """
    Download grokked MLP checkpoints and build the basis-function library.

    Each entry in `spec` becomes one (name, model, cols) tuple. A fresh model
    instance is loaded per entry — sin(x) and sin(v) are separate modules.

    Args:
        repo_id:       HuggingFace repo ID, e.g. "pandevim/cs810"
        load_mlp_fn:   callable(path) -> (nn.Module, int)
                       Use the project's `load_grokked_mlp` here.
        spec:          list of entry dicts; defaults to DEFAULT_SPEC
        keep_filter:   optional callable(spec) -> filtered spec, produced by
                       drop() or keep(); applied BEFORE loading checkpoints.

    Returns:
        List of (name, model, cols) tuples ready for any router constructor.
    """
    from huggingface_hub import hf_hub_download

    if spec is None:
        spec = DEFAULT_SPEC
    if keep_filter is not None:
        spec = keep_filter(spec)

    # Cache checkpoint paths so we only call hf_hub_download once per basis.
    ckpt_cache = {}
    mlp_modules = []

    for entry in spec:
        basis = entry["basis"]
        name = entry_name(entry)

        if basis not in ckpt_cache:
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"phase1/checkpoints/mlp_{basis}.pt",
                )
                ckpt_cache[basis] = path
            except Exception:
                print(f"  Warning: mlp_{basis}.pt not found on HF — skipping all {basis}(*) entries")
                ckpt_cache[basis] = None

        if ckpt_cache[basis] is None:
            continue

        model, _ = load_mlp_fn(ckpt_cache[basis])
        mlp_modules.append((name, model, entry["cols"]))

    return mlp_modules


# ── Filter factories ────────────────────────────────────────────────────────

def drop(basis_names):
    """
    Return a filter that removes all entries whose 'basis' is in basis_names.

    Example:
        mlp_modules = load_library(repo, fn, keep_filter=drop(["sin", "cos"]))
    """
    basis_set = set(basis_names)
    return lambda spec: [e for e in spec if e["basis"] not in basis_set]


def keep(basis_names):
    """
    Return a filter that keeps only entries whose 'basis' is in basis_names.

    Example:
        mlp_modules = load_library(repo, fn, keep_filter=keep(["identity", "add"]))
    """
    basis_set = set(basis_names)
    return lambda spec: [e for e in spec if e["basis"] in basis_set]


def apply_filter(spec, filter_fn):
    """Apply a filter function to a spec list. Returns spec unchanged if filter_fn is None."""
    if filter_fn is None:
        return spec
    return filter_fn(spec)
