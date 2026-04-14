"""
Unified training loop for all Gumbel-Softmax router variants.

All three router classes (StateDepRouter, StateIndepRouter, TopKRouter) share
this single training loop. Differences between experiments live in TrainConfig,
not in duplicated code.

Entropy regularisation
----------------------
The commitment entropy penalty encourages the router to concentrate on a single
basis function (low entropy) rather than spreading probability mass.

Weight schedule: w = entropy_weight_max × max(0, 1 − τ/τ_start)
  - Early (τ high): weight ≈ 0  →  free exploration via Gumbel noise
  - Late  (τ low):  weight → entropy_weight_max  →  force commitment

Set entropy_weight_max=0 to disable (e.g. for StateDepRouter / Exp 1, which
used a different exploration-bonus scheme that is not reproduced here).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainConfig:
    epochs: int = 3000
    lr: float = 3e-3
    tau_start: float = 5.0
    tau_end: float = 0.05
    tau_anneal_epochs: int = 2000
    # Commitment entropy penalty. Ramps up as τ anneals down.
    # Set to 0 to disable (Exp 1 style — state-dep router).
    entropy_weight_max: float = 0.05
    log_every: int = 200


def train_router(router, X_train, dXdt_train, X_val, dXdt_val, cfg, device="cpu"):
    """
    Train a router end-to-end with linear temperature annealing.

    Args:
        router:      any RouterBase subclass
        X_train:     numpy (N_train, state_dim)
        dXdt_train:  numpy (N_train, state_dim)
        X_val:       numpy (N_val, state_dim)
        dXdt_val:    numpy (N_val, state_dim)
        cfg:         TrainConfig
        device:      "cpu" or "cuda"

    Returns:
        history: dict with keys epoch, train_loss, val_loss, temperature
    """
    router = router.to(device)
    X_tr  = torch.from_numpy(X_train.astype(np.float32)).to(device)
    dX_tr = torch.from_numpy(dXdt_train.astype(np.float32)).to(device)
    X_vl  = torch.from_numpy(X_val.astype(np.float32)).to(device)
    dX_vl = torch.from_numpy(dXdt_val.astype(np.float32)).to(device)

    # Optimise only router logits and coefficients — MLPs are frozen.
    optimizer = optim.Adam(
        [p for p in router.parameters() if p.requires_grad],
        lr=cfg.lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_loss": [], "temperature": []}

    for epoch in range(1, cfg.epochs + 1):
        tau = _anneal_tau(epoch, cfg)

        router.train()
        pred, _ = router(X_tr, temperature=tau, hard=True)
        loss = criterion(pred, dX_tr)

        # Commitment entropy penalty (disabled when entropy_weight_max=0).
        entropy_weight = cfg.entropy_weight_max * max(0.0, 1.0 - tau / cfg.tau_start)
        if entropy_weight > 0:
            loss = loss + entropy_weight * router.logit_entropy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % cfg.log_every == 0 or epoch == 1:
            router.eval()
            with torch.no_grad():
                val_pred, _ = router(X_vl, temperature=tau, hard=True)
                val_loss = criterion(val_pred, dX_vl)

            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss.item())
            history["temperature"].append(tau)

            print(
                f"  Epoch {epoch:>5d} | τ={tau:.3f} | "
                f"Train: {loss.item():.6f} | Val: {val_loss.item():.6f}"
            )

    return history


def _anneal_tau(epoch, cfg):
    """Linear temperature decay from tau_start to tau_end over tau_anneal_epochs."""
    if epoch <= cfg.tau_anneal_epochs:
        return cfg.tau_start - (cfg.tau_start - cfg.tau_end) * (epoch / cfg.tau_anneal_epochs)
    return cfg.tau_end
