import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from huggingface_hub import hf_hub_download

from phase1_train_mlp import GrokMLP

# ─── MLP Library ─────────────────────────────────────────────────────────────

class MLPLibraryTerm:
    """A single term in the neural SINDy library."""

    def __init__(self, name, model, input_dim, input_columns):
        """
        Args:
            name: human-readable name (e.g., "cos(x)")
            model: trained GrokMLP
            input_dim: how many inputs the MLP expects (1 or 2)
            input_columns: which columns of the state X to feed in.
                           For unary: [0] or [1]
                           For binary: [0, 1]
        """
        self.name = name
        self.model = model
        self.input_dim = input_dim
        self.input_columns = input_columns

    def evaluate(self, X):
        """
        Evaluate this MLP on state data X of shape (N, state_dim).
        Returns (N, 1) array.
        """
        self.model.eval()
        with torch.no_grad():
            # Select relevant columns
            x_in = torch.from_numpy(
                X[:, self.input_columns].astype(np.float32)
            )
            out = self.model(x_in).numpy()
        return out


class NeuralSINDyLibrary:
    """
    Library of grokked MLPs for SINDy-style system identification.

    Given state data X (shape N x state_dim), evaluates every MLP
    in the library to construct the feature matrix Θ.
    """

    def __init__(self):
        self.terms = []

    def add_term(self, name, model, input_dim, input_columns):
        term = MLPLibraryTerm(name, model, input_dim, input_columns)
        self.terms.append(term)
        return self

    def build_theta(self, X):
        """
        Construct the library matrix Θ of shape (N, n_terms).

        Each column is the output of one MLP evaluated on the state data.
        """
        columns = []
        for term in self.terms:
            col = term.evaluate(X)  # (N, 1)
            columns.append(col)

        Theta = np.hstack(columns)  # (N, n_terms)
        return Theta

    def get_term_names(self):
        return [t.name for t in self.terms]


# ─── Sparse Regression (STLSQ) ──────────────────────────────────────────────

def stlsq(Theta, dXdt_col, threshold=0.05, max_iter=20, alpha=0.01):
    """
    Sequentially Thresholded Least Squares (STLSQ).

    This is the original SINDy optimizer:
    1. Solve least squares: ξ = (Θ^T Θ)^-1 Θ^T Ẋ
    2. Zero out coefficients below threshold
    3. Repeat with remaining terms until stable

    Args:
        Theta: library matrix (N, n_terms)
        dXdt_col: single column of derivatives (N,)
        threshold: sparsity cutoff
        max_iter: max iterations
        alpha: ridge regularization

    Returns:
        xi: sparse coefficient vector (n_terms,)
    """
    n_terms = Theta.shape[1]
    xi = np.zeros(n_terms)

    # Initial ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Theta, dXdt_col)
    xi = ridge.coef_.copy()

    for iteration in range(max_iter):
        # Threshold small coefficients
        small_idx = np.abs(xi) < threshold
        xi[small_idx] = 0.0

        # Re-fit with remaining terms
        big_idx = ~small_idx
        if not np.any(big_idx):
            break

        ridge_sub = Ridge(alpha=alpha, fit_intercept=False)
        ridge_sub.fit(Theta[:, big_idx], dXdt_col)
        xi[big_idx] = ridge_sub.coef_

        # Check convergence
        if np.all(np.abs(xi[small_idx]) == 0):
            # Check if any newly fitted coefficient is below threshold
            if np.all(np.abs(xi[big_idx]) >= threshold):
                break

    return xi


# ─── Load Grokked Models ────────────────────────────────────────────────────

def load_grokked_mlp(checkpoint_path):
    """Load a trained GrokMLP from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GrokMLP(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        activation=ckpt["activation"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["input_dim"]


def build_default_library(repo_id="pandevim/cs810"):
    """
    Build a NeuralSINDyLibrary from all trained MLP checkpoints.

    For a 2D state [x, v], we create terms:
        - Unary MLPs applied to x (column 0)
        - Unary MLPs applied to v (column 1)
        - Binary MLPs applied to (x, v)
        - Constant bias term
    """
    library = NeuralSINDyLibrary()

    # Load all models
    unary_names = ["identity", "cos", "sin"]
    binary_names = ["add", "mul"]

    for name in unary_names:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=f"phase1/checkpoints/mlp_{name}.pt")
        except Exception:
            print(f"  ⚠ Checkpoint not found on HF: mlp_{name}.pt, skipping")
            continue

        model, input_dim = load_grokked_mlp(path)

        # Apply to each state variable
        library.add_term(f"{name}(x)", model, input_dim, [0])
        library.add_term(f"{name}(v)", model, input_dim, [1])

    for name in binary_names:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=f"phase1/checkpoints/mlp_{name}.pt")
        except Exception:
            print(f"  ⚠ Checkpoint not found on HF: mlp_{name}.pt, skipping")
            continue

        model, input_dim = load_grokked_mlp(path)
        library.add_term(f"{name}(x,v)", model, input_dim, [0, 1])

    return library


# ─── Discovery ───────────────────────────────────────────────────────────────

def discover_equations(X, dXdt, library, threshold=0.05, alpha=0.01):
    """
    Run Neural SINDy discovery.

    Args:
        X: state data (N, state_dim)
        dXdt: derivatives (N, state_dim)
        library: NeuralSINDyLibrary instance
        threshold: STLSQ sparsity threshold
        alpha: ridge regularization

    Returns:
        xi: coefficient matrix (n_terms, state_dim)
        term_names: list of term names
    """
    print("\n  Building library matrix Θ...")
    Theta = library.build_theta(X)
    term_names = library.get_term_names()
    n_terms = len(term_names)
    state_dim = dXdt.shape[1]

    print(f"  Θ shape: {Theta.shape} ({len(term_names)} library terms)")
    print(f"  Library terms: {term_names}\n")

    xi = np.zeros((n_terms, state_dim))

    state_labels = ["ẋ (dx/dt)", "v̇ (dv/dt)"]

    for col in range(state_dim):
        print(f"  ─── Discovering equation for {state_labels[col]} ───")
        xi[:, col] = stlsq(Theta, dXdt[:, col], threshold=threshold, alpha=alpha)

        # Print discovered equation
        active = np.abs(xi[:, col]) > 1e-10
        terms = []
        for i in range(n_terms):
            if active[i]:
                coeff = xi[i, col]
                terms.append(f"{coeff:+.4f}·{term_names[i]}")

        eq_str = " ".join(terms) if terms else "0"
        print(f"  {state_labels[col]} = {eq_str}\n")

    return xi, term_names