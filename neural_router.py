"""
Gumbel-Softmax Neural Router for MLP Library Selection.

This is the end-to-end differentiable variant of Neural SINDy.
Instead of using STLSQ sparse regression (a post-hoc, non-gradient method),
we train a "router" network that learns to select which grokked MLP
to apply at each state, using Gumbel-Softmax for differentiable discrete choices.

Architecture:
    State [x, v] → Router Network → Gumbel-Softmax → Gate (one-hot) → Σ(gate_i · MLP_i(state))

The router learns WHICH MLP explains each component of the dynamics,
while the grokked MLPs provide the actual function approximations.

Key technique: Straight-Through Estimator (STE)
    Forward pass: hard one-hot selection (only 1 MLP computes)
    Backward pass: smooth Gumbel-Softmax gradients flow through all MLPs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path

from grok_mlps import GrokMLP
from neural_sindy import load_grokked_mlp


# ─── Gumbel-Softmax ──────────────────────────────────────────────────────────

def gumbel_softmax(logits, temperature=1.0, hard=True):
    """
    Sample from the Gumbel-Softmax distribution.

    Args:
        logits: (batch, n_choices) unnormalized log-probabilities
        temperature: τ — controls sharpness. High=soft, Low=hard.
        hard: if True, use Straight-Through Estimator (STE)
              Forward: hard one-hot. Backward: soft gradients.

    Returns:
        (batch, n_choices) — one-hot if hard=True, soft probabilities otherwise
    """
    # Sample Gumbel noise: g_i = -log(-log(u_i)), u_i ~ Uniform(0,1)
    gumbel_noise = -torch.log(-torch.log(
        torch.rand_like(logits).clamp(min=1e-10)
    ))

    # Add noise to logits and apply temperature-scaled softmax
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-Through Estimator
        # Forward: hard one-hot (argmax)
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # Backward: pretend we used y_soft (gradient flows through)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


# ─── Router Network ──────────────────────────────────────────────────────────

class GumbelRouter(nn.Module):
    """
    A router that learns to select which grokked MLP to apply for each
    component of the dynamics.

    For a system with state_dim=2 (x, v), we learn TWO routing decisions:
        - Which MLP combination explains dx/dt?
        - Which MLP combination explains dv/dt?

    Each routing decision selects from the full MLP library.
    """

    def __init__(self, state_dim, mlp_modules, mlp_configs, router_hidden=64):
        """
        Args:
            state_dim: dimension of the state space (e.g., 2 for [x, v])
            mlp_modules: list of (name, model, input_columns) tuples
            mlp_configs: list of dicts with 'input_dim' for each MLP
            router_hidden: hidden dimension for the router network
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_mlps = len(mlp_modules)
        self.mlp_names = [m[0] for m in mlp_modules]
        self.mlp_models = nn.ModuleList([m[1] for m in mlp_modules])
        self.input_columns = [m[2] for m in mlp_modules]

        # Freeze grokked MLPs — we don't want to un-grok them!
        for model in self.mlp_models:
            for param in model.parameters():
                param.requires_grad = False

        # One router per state derivative (dx/dt, dv/dt, ...)
        # Each router outputs logits over the MLP library
        self.routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, router_hidden),
                nn.ReLU(),
                nn.Linear(router_hidden, router_hidden),
                nn.ReLU(),
                nn.Linear(router_hidden, self.n_mlps),
            )
            for _ in range(state_dim)
        ])

        # Learnable coefficients for each selected MLP
        # (the router picks WHICH MLP, the coefficient scales HOW MUCH)
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_mlps))
            for _ in range(state_dim)
        ])

    def forward(self, X, temperature=1.0, hard=True):
        """
        Forward pass: route state data through selected MLPs.

        Args:
            X: state tensor (batch, state_dim)
            temperature: Gumbel-Softmax temperature
            hard: use Straight-Through Estimator

        Returns:
            dXdt_pred: predicted derivatives (batch, state_dim)
            gates: routing decisions (state_dim, batch, n_mlps)
        """
        batch_size = X.shape[0]

        # Evaluate ALL MLPs on the state data
        # (we need all outputs to compute the weighted sum)
        mlp_outputs = []  # will be (n_mlps, batch, 1)
        for i, (model, cols) in enumerate(zip(self.mlp_models, self.input_columns)):
            x_in = X[:, cols]  # select input columns
            out = model(x_in)  # (batch, 1)
            mlp_outputs.append(out)

        # Stack: (batch, n_mlps)
        mlp_out_matrix = torch.cat(mlp_outputs, dim=1)

        # For each state derivative, compute routing + output
        dXdt_pred = []
        all_gates = []

        for d in range(self.state_dim):
            # Router produces logits over MLPs
            logits = self.routers[d](X)  # (batch, n_mlps)

            # Gumbel-Softmax: differentiable discrete selection
            gate = gumbel_softmax(logits, temperature=temperature, hard=hard)
            all_gates.append(gate)

            # Weighted sum: gate selects MLP, coefficients scale it
            # gate: (batch, n_mlps), coefficients: (n_mlps,)
            # mlp_out_matrix: (batch, n_mlps)
            coeffs = self.coefficients[d]  # (n_mlps,)
            weighted = (gate * coeffs.unsqueeze(0)) * mlp_out_matrix  # (batch, n_mlps)
            dxdt_d = weighted.sum(dim=1, keepdim=True)  # (batch, 1)
            dXdt_pred.append(dxdt_d)

        dXdt_pred = torch.cat(dXdt_pred, dim=1)  # (batch, state_dim)
        all_gates = torch.stack(all_gates, dim=0)  # (state_dim, batch, n_mlps)

        return dXdt_pred, all_gates

    def get_routing_summary(self, X, temperature=0.01):
        """Get a summary of which MLPs are selected for which derivatives."""
        self.eval()
        with torch.no_grad():
            _, gates = self.forward(X, temperature=temperature, hard=True)

        # Average gate activations across batch
        avg_gates = gates.mean(dim=1)  # (state_dim, n_mlps)

        summary = {}
        state_labels = ["dx/dt", "dv/dt"]
        for d in range(self.state_dim):
            selected = []
            for i in range(self.n_mlps):
                activation = avg_gates[d, i].item()
                coeff = self.coefficients[d][i].item()
                if activation > 0.01:  # at least 1% selection
                    selected.append({
                        "name": self.mlp_names[i],
                        "activation": activation,
                        "coefficient": coeff,
                    })
            # Sort by activation
            selected.sort(key=lambda x: -x["activation"])
            summary[state_labels[d]] = selected

        return summary


# ─── Training ────────────────────────────────────────────────────────────────

def train_router(
    router, X_train, dXdt_train, X_val, dXdt_val,
    epochs=2000, lr=3e-3,
    tau_start=5.0, tau_end=0.1, tau_anneal_epochs=1500,
    log_every=100, device="cpu",
):
    """
    Train the Gumbel-Softmax router end-to-end.

    Temperature annealing schedule:
        - Start with high τ (soft selection, easy gradients)
        - Linearly anneal to low τ (hard selection, discrete choices)
    """
    router = router.to(device)
    X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
    dXdt_train = torch.from_numpy(dXdt_train.astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_val.astype(np.float32)).to(device)
    dXdt_val = torch.from_numpy(dXdt_val.astype(np.float32)).to(device)

    # Only optimize router parameters + coefficients (MLPs are frozen)
    optimizer = optim.Adam(
        list(router.routers.parameters()) + list(router.coefficients.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "temperature": [], "epoch": []}

    for epoch in range(1, epochs + 1):
        # Temperature annealing: linear decay
        if epoch <= tau_anneal_epochs:
            tau = tau_start - (tau_start - tau_end) * (epoch / tau_anneal_epochs)
        else:
            tau = tau_end

        # ── Train ──
        router.train()
        pred, gates = router(X_train, temperature=tau, hard=True)
        loss = criterion(pred, dXdt_train)

        # Optional: add entropy regularization to encourage exploration early on
        if tau > 1.0:
            gate_probs = gates.mean(dim=1)  # (state_dim, n_mlps)
            entropy = -(gate_probs * (gate_probs + 1e-10).log()).sum()
            loss = loss - 0.01 * entropy  # encourage exploration

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Log ──
        if epoch % log_every == 0 or epoch == 1:
            router.eval()
            with torch.no_grad():
                val_pred, _ = router(X_val, temperature=tau, hard=True)
                val_loss = criterion(val_pred, dXdt_val)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["temperature"].append(tau)
            history["epoch"].append(epoch)

            print(
                f"  Epoch {epoch:>5d} | τ={tau:.3f} | "
                f"Train: {loss.item():.6f} | Val: {val_loss.item():.6f}"
            )

    return history


# ─── Build Router from Checkpoints ──────────────────────────────────────────

def build_router_from_checkpoints(checkpoint_dir="checkpoints", state_dim=2):
    """
    Load grokked MLPs and construct a GumbelRouter.

    For state [x, v] with state_dim=2, we create MLP terms:
        - Unary MLPs applied to x (column 0)
        - Unary MLPs applied to v (column 1)
        - Binary MLPs applied to (x, v)
    """
    ckpt_dir = Path(checkpoint_dir)
    mlp_modules = []  # list of (name, model, input_columns)
    mlp_configs = []

    unary_names = ["identity", "cos", "sin"]
    binary_names = ["add", "mul"]

    for name in unary_names:
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            print(f"  ⚠ {path} not found, skipping")
            continue
        model, input_dim = load_grokked_mlp(path)
        # Apply to each state variable
        mlp_modules.append((f"{name}(x)", model, [0]))
        mlp_configs.append({"input_dim": input_dim})

        # Clone model for second variable (separate instance)
        model2, _ = load_grokked_mlp(path)
        mlp_modules.append((f"{name}(v)", model2, [1]))
        mlp_configs.append({"input_dim": input_dim})

    for name in binary_names:
        path = ckpt_dir / f"mlp_{name}.pt"
        if not path.exists():
            print(f"  ⚠ {path} not found, skipping")
            continue
        model, input_dim = load_grokked_mlp(path)
        mlp_modules.append((f"{name}(x,v)", model, [0, 1]))
        mlp_configs.append({"input_dim": input_dim})

    router = GumbelRouter(
        state_dim=state_dim,
        mlp_modules=mlp_modules,
        mlp_configs=mlp_configs,
        router_hidden=64,
    )

    return router
