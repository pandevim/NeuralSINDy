"""
Router classes for the Gumbel-Softmax Neural SINDy pipeline.

All routers share the same constructor contract:
    mlp_modules: list of (name, model, cols) tuples from library.load_library

Three concrete classes, corresponding to the three experiments:

    StateDepRouter   (Exp 1) — state-dependent MLP produces per-sample logits.
    StateIndepRouter (Exp 2) — one global logit vector per derivative,
                               broadcast across the batch. SINDy-aligned.
    TopKRouter       (Exp 3) — state-independent + picks k distinct basis
                               functions via masked Gumbel sampling +
                               a complexity prior in logit space.

All routers implement:
    forward(X, temperature, hard) → (dXdt_pred, gates)
    get_routing_summary(X, temperature) → summary dict (see below)
    logit_entropy() → scalar tensor  (for commitment penalty in training)
    to_spec(summary) → Phase-6-compatible spec dict

Summary format (returned by get_routing_summary)
-------------------------------------------------
    {
        "dx/dt": [{"name": "identity(v)", "activation": 0.997, "coefficient": 1.002}, ...],
        "dv/dt": [{"name": "identity(x)", "activation": 0.993, "coefficient": -1.005}, ...],
    }

Phase-6 spec format (returned by spec_from_summary)
-----------------------------------------------------
    {
        "derivs": [
            {"name": "dx/dt", "terms": [{"basis": "identity(v)", "coefficient": 1.0, "activation": 0.997}]},
            {"name": "dv/dt", "terms": [{"basis": "identity(x)", "coefficient": -1.0, "activation": 0.993}]},
        ]
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Gumbel-Softmax primitive ─────────────────────────────────────────────────

def gumbel_softmax(logits, temperature=1.0, hard=True):
    """
    Gumbel-Softmax with Straight-Through Estimator (STE).

    Args:
        logits:      (batch, n_choices)
        temperature: τ — high = soft/exploratory, low = hard/discrete
        hard:        if True, forward is one-hot, backward uses soft gradients

    Returns:
        (batch, n_choices) — one-hot in forward pass when hard=True
    """
    gumbel_noise = -torch.log(-torch.log(
        torch.rand_like(logits).clamp(min=1e-10)
    ))
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # STE
    return y_soft


# ── Base class ───────────────────────────────────────────────────────────────

class RouterBase(nn.Module):
    """Common interface and shared machinery for all routers."""

    DERIV_LABELS = ["dx/dt", "dv/dt"]

    def __init__(self, state_dim, mlp_modules):
        super().__init__()
        self.state_dim = state_dim
        self.n_mlps = len(mlp_modules)
        self.mlp_names = [m[0] for m in mlp_modules]
        self.mlp_models = nn.ModuleList([m[1] for m in mlp_modules])
        self.input_columns = [m[2] for m in mlp_modules]

        # Grokked MLPs are frozen — we learn which ones to use, not the MLPs themselves.
        for model in self.mlp_models:
            for p in model.parameters():
                p.requires_grad = False

    def _eval_library(self, X):
        """Evaluate every MLP and return (batch, n_mlps) output matrix."""
        return torch.cat(
            [model(X[:, cols]) for model, cols in zip(self.mlp_models, self.input_columns)],
            dim=1,
        )

    def get_routing_summary(self, X, temperature=0.01):
        raise NotImplementedError

    def logit_entropy(self):
        """
        Commitment entropy for the training loop's regularisation term.
        Higher entropy = more spread = less committed.
        Training minimises this (penalises spread) as temperature anneals.
        """
        raise NotImplementedError


# ── Experiment 1: State-dependent router ─────────────────────────────────────

class StateDepRouter(RouterBase):
    """
    Exp 1 router: a small MLP maps state [x, v] to per-derivative routing logits.
    Gate probabilities vary per sample — the router is NOT globally consistent.

    This design violates the SINDy assumption (single global term governs the
    dynamics everywhere), which is why Exp 2 moved to state-independent routing.
    Kept here for ablation and historical comparison.
    """

    def __init__(self, state_dim, mlp_modules, router_hidden=64):
        super().__init__(state_dim, mlp_modules)
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
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_mlps)) for _ in range(state_dim)
        ])

    def forward(self, X, temperature=1.0, hard=True):
        """
        Returns:
            dXdt_pred: (batch, state_dim)
            gates:     (state_dim, batch, n_mlps)
        """
        mlp_out = self._eval_library(X)  # (batch, n_mlps)
        dXdt_pred, all_gates = [], []

        for d in range(self.state_dim):
            logits = self.routers[d](X)  # (batch, n_mlps)
            gate = gumbel_softmax(logits, temperature=temperature, hard=hard)
            all_gates.append(gate)
            weighted = (gate * self.coefficients[d].unsqueeze(0)) * mlp_out
            dXdt_pred.append(weighted.sum(dim=1, keepdim=True))

        return (torch.cat(dXdt_pred, dim=1),
                torch.stack(all_gates, dim=0))

    def get_routing_summary(self, X, temperature=0.01):
        self.eval()
        with torch.no_grad():
            _, gates = self.forward(X, temperature=temperature, hard=True)
        avg_gates = gates.mean(dim=1)  # (state_dim, n_mlps)
        return _make_summary(self, avg_gates)

    def logit_entropy(self):
        # No global logit vector to compute entropy over; return zero.
        # (Exp 1 used a gate-based exploration bonus instead — see training.py.)
        return torch.tensor(0.0)


# ── Experiment 2: State-independent router ────────────────────────────────────

class StateIndepRouter(RouterBase):
    """
    Exp 2 router: one learnable logit vector per derivative, broadcast to the
    full batch. Every sample in the batch gets the same routing decision (before
    Gumbel noise), matching SINDy's assumption that dynamics are globally uniform.

    Param count: 2 × n_mlps logits + 2 × n_mlps coefficients = 32 for 8 MLPs.
    (vs ~9 760 for StateDepRouter with router_hidden=64).
    """

    def __init__(self, state_dim, mlp_modules):
        super().__init__(state_dim, mlp_modules)
        self.routers = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_mlps)) for _ in range(state_dim)
        ])
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_mlps)) for _ in range(state_dim)
        ])

    def forward(self, X, temperature=1.0, hard=True):
        """
        Returns:
            dXdt_pred: (batch, state_dim)
            gates:     (state_dim, batch, n_mlps)
        """
        batch_size = X.shape[0]
        mlp_out = self._eval_library(X)
        dXdt_pred, all_gates = [], []

        for d in range(self.state_dim):
            # Broadcast global logits so each sample still draws its own Gumbel noise.
            logits = self.routers[d].unsqueeze(0).expand(batch_size, -1)
            gate = gumbel_softmax(logits, temperature=temperature, hard=hard)
            all_gates.append(gate)
            weighted = (gate * self.coefficients[d].unsqueeze(0)) * mlp_out
            dXdt_pred.append(weighted.sum(dim=1, keepdim=True))

        return (torch.cat(dXdt_pred, dim=1),
                torch.stack(all_gates, dim=0))

    def get_routing_summary(self, X, temperature=0.01):
        self.eval()
        with torch.no_grad():
            _, gates = self.forward(X, temperature=temperature, hard=True)
        avg_gates = gates.mean(dim=1)
        return _make_summary(self, avg_gates)

    def logit_entropy(self):
        ent = torch.tensor(0.0)
        for d in range(self.state_dim):
            probs = F.softmax(self.routers[d], dim=-1)
            ent = ent + -(probs * (probs + 1e-10).log()).sum()
        return ent


# ── Experiment 3: Top-k router with complexity prior ─────────────────────────

class TopKRouter(RouterBase):
    """
    Exp 3 router: state-independent + selects k distinct basis functions per
    derivative via iterative masked Gumbel-Softmax (sampling without replacement).

    Why k > 1: the true v̇ = -k·x - c·v needs TWO active terms. One-hot
    routing (Exp 2) can't express it.

    Complexity prior: adds a bias in logit space favouring simpler basis
    functions (identity > sin/cos > add/mul). See build_complexity_prior().
    """

    def __init__(self, state_dim, mlp_modules, k=2, complexity_prior=None):
        super().__init__(state_dim, mlp_modules)
        self.k = k
        self.routers = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_mlps)) for _ in range(state_dim)
        ])
        # One coefficient per (derivative, slot, mlp).
        self.coefficients = nn.ParameterList([
            nn.Parameter(torch.zeros(k, self.n_mlps)) for _ in range(state_dim)
        ])
        if complexity_prior is None:
            complexity_prior = torch.zeros(self.n_mlps)
        self.register_buffer("complexity_prior", complexity_prior.float())

    def forward(self, X, temperature=1.0, hard=True):
        """
        Returns:
            dXdt_pred: (batch, state_dim)
            gates:     (state_dim, k, batch, n_mlps)
        """
        batch_size = X.shape[0]
        mlp_out = self._eval_library(X)
        dXdt_pred, all_gates = [], []

        for d in range(self.state_dim):
            base_logits = (self.routers[d] + self.complexity_prior).unsqueeze(0).expand(batch_size, -1)
            mask = torch.zeros_like(base_logits)
            slot_gates = []
            dxdt_d = torch.zeros(batch_size, 1, device=X.device)

            for j in range(self.k):
                gate_j = gumbel_softmax(base_logits + mask, temperature=temperature, hard=hard)
                slot_gates.append(gate_j)
                # Exclude this pick from future slots (detached so mask doesn't carry gradient).
                mask = mask + (-1e9) * gate_j.detach()
                weighted = (gate_j * self.coefficients[d][j].unsqueeze(0)) * mlp_out
                dxdt_d = dxdt_d + weighted.sum(dim=1, keepdim=True)

            all_gates.append(torch.stack(slot_gates, dim=0))
            dXdt_pred.append(dxdt_d)

        return (torch.cat(dXdt_pred, dim=1),
                torch.stack(all_gates, dim=0))

    def get_routing_summary(self, X, temperature=0.01):
        self.eval()
        with torch.no_grad():
            _, gates = self.forward(X, temperature=temperature, hard=True)
        # gates: (state_dim, k, batch, n_mlps) — collapse slots and batch.
        avg = gates.mean(dim=2)  # (state_dim, k, n_mlps)

        device = next(self.parameters()).device
        summary = {}
        for d in range(self.state_dim):
            eff_coeff = torch.zeros(self.n_mlps, device=device)
            total_act = torch.zeros(self.n_mlps, device=device)
            for j in range(self.k):
                act_j = avg[d, j]
                coeff_j = self.coefficients[d][j].detach()
                eff_coeff += act_j * coeff_j
                total_act += act_j

            selected = [
                {
                    "name": self.mlp_names[i],
                    "activation": total_act[i].item(),
                    "coefficient": eff_coeff[i].item(),
                }
                for i in range(self.n_mlps)
                if total_act[i].item() > 0.01
            ]
            selected.sort(key=lambda s: -s["activation"])
            summary[self.DERIV_LABELS[d]] = selected
        return summary

    def logit_entropy(self):
        """
        Conditional slot entropy.

        Exp 3's original version penalised the base distribution, which drove it
        one-hot and left slot 2 with a flat conditional — preventing commitment.

        This version iterates through slots exactly as forward() does, computing
        H(slot_j | slot_0..j-1 chosen) and summing across slots and derivatives.
        Each slot's conditional distribution is penalised independently, so every
        slot commits rather than just the first.

        Mask advance uses detached argmax (the expected winner at low τ), mirroring
        the detached gate in forward() so no gradient flows through the mask.
        """
        ent = torch.tensor(0.0, device=self.complexity_prior.device)
        for d in range(self.state_dim):
            base = self.routers[d] + self.complexity_prior  # (n_mlps,)
            mask = torch.zeros_like(base)
            for _ in range(self.k):
                logits = base + mask
                probs = F.softmax(logits, dim=-1)
                ent = ent + -(probs * (probs + 1e-10).log()).sum()
                winner = probs.detach().argmax()
                mask = mask.clone()
                mask[winner] = -1e9
        return ent


# ── Utilities ────────────────────────────────────────────────────────────────

def build_complexity_prior(mlp_names, alpha=1.0):
    """
    Occam's-razor bias in logit space.

    identity: +alpha  (simplest)
    sin/cos:  0
    add/mul:  -alpha  (compositional, penalised)

    alpha=1.0 nudges probabilities by a factor of e in softmax space.
    """
    prior = torch.zeros(len(mlp_names))
    for i, name in enumerate(mlp_names):
        root = name.split("(")[0]
        if root == "identity":
            prior[i] = +alpha
        elif root in ("sin", "cos"):
            prior[i] = 0.0
        elif root in ("add", "mul"):
            prior[i] = -alpha
    return prior


def spec_from_summary(summary):
    """
    Convert a routing summary dict to the Phase-6 structured spec.

    Phase 6 (symbolic distillation) consumes this format regardless of
    which router variant produced it.
    """
    return {
        "derivs": [
            {
                "name": deriv_name,
                "terms": [
                    {
                        "basis": term["name"],
                        "coefficient": term["coefficient"],
                        "activation": term["activation"],
                    }
                    for term in terms
                ],
            }
            for deriv_name, terms in summary.items()
        ]
    }


def _make_summary(router, avg_gates):
    """Build summary dict from avg_gates tensor (state_dim, n_mlps)."""
    summary = {}
    for d in range(router.state_dim):
        selected = [
            {
                "name": router.mlp_names[i],
                "activation": avg_gates[d, i].item(),
                "coefficient": router.coefficients[d][i].item(),
            }
            for i in range(router.n_mlps)
            if avg_gates[d, i].item() > 0.01
        ]
        selected.sort(key=lambda s: -s["activation"])
        summary[router.DERIV_LABELS[d]] = selected
    return summary
