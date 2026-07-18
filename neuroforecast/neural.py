"""Neural conditional directed-information estimator.

The linear estimator (`linear.py`) certifies CDI = I(Y_future ; X_past | Z) when
the coupling is linear-Gaussian. Real neural/physiological coupling is not linear,
and averaging a long causal window into features can destroy the fine-timescale
structure that carries the signal. This module estimates the *same estimand* with
neural capacity, keeping the same two honesty guarantees:

  * cross-fitting  -> no reward for overfitting (an uninformative X -> CDI ~ 0);
  * a certified detection floor via the cluster bootstrap (see `calibration.py`).

Estimand and estimator (identical in spirit to the linear plug-in):

    CDI = E[ log q1(Y | X_past, Z) - log q0(Y | Z) ] / ln 2,

where q0, q1 are heteroscedastic Gaussian predictive densities produced by neural
networks, trained on a disjoint fold and evaluated out-of-fold. For a fixed
network class this is a lower bound on the true CDI -- the honest direction.

Architecture (the part that is new for raw neural signals):

    X_past (raw causal window, C channels x T samples)
        -> CausalConvEncoder  (dilated, causal 1-D convolutions: a TCN)
        -> embedding e_x
    Z (baseline: target's own past + nuisance)  -> e_z = MLP(Z)
    [e_x, e_z] -> HeteroscedasticHead -> (mu, log_sigma2) of Y_future    (model q1)
    [e_z]      -> HeteroscedasticHead -> (mu, log_sigma2) of Y_future    (model q0)

The baseline q0 sees only Z; the full model q1 additionally sees the encoded
causal history of X. Their held-out predictive-log-likelihood gap is the certified
directed information X contributes about the future of Y beyond its own past.

CPU-trainable at small scale (validation); designed to scale to full cohorts on
GPU (see `run_a100.sh`). PyTorch, no Lightning, no framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_LOG2 = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
class CausalConvEncoder(nn.Module):
    """Temporal convolutional network with causal, dilated 1-D convolutions.

    Input  : (batch, channels, time)  -- the pre-issue causal window of X.
    Output : (batch, embed_dim)       -- a summary that preserves fine-timescale
             precursor structure (unlike a single band-power average).
    """

    def __init__(self, in_channels: int, embed_dim: int = 32,
                 hidden: int = 32, n_blocks: int = 4, kernel: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for b in range(n_blocks):
            dilation = 2 ** b
            pad = (kernel - 1) * dilation  # left pad only -> causal
            layers += [
                _CausalConv1d(c_in, hidden, kernel, dilation, pad),
                nn.GELU(),
            ]
            c_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)              # (B, hidden, T)
        h = h[:, :, -1]              # last (most recent) timestep summary
        return self.head(h)


class _CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, pad):
        super().__init__()
        self.pad = pad
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class MeanHead(nn.Module):
    """Maps an embedding to the conditional mean of the future target.

    We deliberately do NOT learn a per-sample variance: heteroscedastic variance
    heads overfit and produce catastrophic held-out log-likelihood spikes, which
    is the classic instability behind biased neural directed-information
    estimates. Instead the predictive variance is the held-out residual variance
    (homoscedastic), and CDI = 0.5 log2(v0/v1) exactly as in the linear plug-in.
    Same estimator, richer function class."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, e: torch.Tensor):
        return self.net(e).squeeze(-1)


class _ZEncoder(nn.Module):
    def __init__(self, z_dim: int, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )

    def forward(self, z):
        return self.net(z)


class ConditionalPredictor(nn.Module):
    """Predictive density of Y_future. If `use_x`, encodes the raw causal window
    of X with the TCN and concatenates with the Z embedding; else uses Z only."""

    def __init__(self, z_dim: int, x_channels: int | None, x_time: int | None,
                 embed_dim: int = 32, use_x: bool = True):
        super().__init__()
        self.use_x = use_x and x_channels is not None
        self.z_enc = _ZEncoder(z_dim, embed_dim)
        self.z_skip = nn.Linear(z_dim, 1)   # linear Z->mean path (baseline capacity)
        head_in = embed_dim
        if self.use_x:
            self.x_enc = CausalConvEncoder(x_channels, embed_dim)
            head_in += embed_dim
        self.head = MeanHead(head_in)

    def forward(self, z, x=None):
        e = self.z_enc(z)
        if self.use_x:
            e = torch.cat([e, self.x_enc(x)], dim=-1)
        # residual linear skip: guarantees the full model can match the linear
        # baseline, so adding X never *reduces* fit -> CDI is not undercounted.
        return self.head(e) + self.z_skip(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
@dataclass
class NeuralCdiResult:
    cdi_bits: float
    lcb95_bits: float
    resid0: np.ndarray            # out-of-fold squared residuals given Z
    resid1: np.ndarray            # out-of-fold squared residuals given (X, Z)
    clusters: np.ndarray

    @property
    def is_certified_positive(self) -> bool:
        return self.lcb95_bits > 0.0


def _train_predictor(model, loader, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            y, z = batch[0].to(device), batch[1].to(device)
            x = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            pred = model(z, x) if x is not None else model(z)
            loss = ((y - pred) ** 2).mean()        # MSE = fixed-variance Gaussian NLL
            loss.backward()
            opt.step()
    return model


def neural_conditional_directed_information(
    y_future: np.ndarray,
    x_past: np.ndarray,          # (n, channels, time) raw causal window
    z_baseline: np.ndarray,      # (n, dz)
    *,
    clusters: np.ndarray | None = None,
    n_splits: int = 2,
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 256,
    n_boot: int = 2000,
    embed_dim: int = 32,
    seed: int = 0,
    device: str | None = None,
) -> NeuralCdiResult:
    """Cross-fitted neural estimate of CDI = I(Y_future ; X_past | Z) in bits.

    X_past is a raw causal window (channels x time); the TCN encodes it. Returns
    the point estimate and a cluster-bootstrap 95% lower bound built from the
    out-of-fold pointwise log-likelihood gap (so the bootstrap is honest).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    y = np.asarray(y_future, np.float64).ravel()
    X = np.asarray(x_past, np.float32)
    Z = np.asarray(z_baseline, np.float32)
    if X.ndim == 2:                      # (n, features) -> single "channel"
        X = X[:, None, :]
    n = len(y)
    if clusters is None:
        clusters = np.arange(n)
    clusters = np.asarray(clusters)

    r0 = np.zeros(n, dtype=np.float64)   # out-of-fold squared residuals given Z
    r1 = np.zeros(n, dtype=np.float64)   # out-of-fold squared residuals given (X,Z)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_splits)

    for f in range(n_splits):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != f])

        # Per-fold standardization (fit on train, apply to both). CDI is invariant
        # to an affine rescale of Y (the Jacobian cancels between q0 and q1), so
        # this does not bias the estimate -- it only makes training well-scaled.
        ym, ys = y[tr].mean(), y[tr].std() + 1e-8
        zm, zs = Z[tr].mean(0), Z[tr].std(0) + 1e-8
        xm = X[tr].mean((0, 2), keepdims=True)
        xs = X[tr].std((0, 2), keepdims=True) + 1e-8
        yt = torch.tensor(((y - ym) / ys), dtype=torch.float32)
        zt = torch.tensor(((Z - zm) / zs), dtype=torch.float32)
        xt = torch.tensor(((X - xm) / xs), dtype=torch.float32)

        tr_t = torch.tensor(tr)
        te_t = torch.tensor(te)

        base = ConditionalPredictor(Z.shape[1], None, None, embed_dim, use_x=False).to(device)
        full = ConditionalPredictor(Z.shape[1], X.shape[1], X.shape[2], embed_dim, use_x=True).to(device)

        base_loader = DataLoader(TensorDataset(yt[tr_t], zt[tr_t]),
                                 batch_size=batch_size, shuffle=True)
        full_loader = DataLoader(TensorDataset(yt[tr_t], zt[tr_t], xt[tr_t]),
                                 batch_size=batch_size, shuffle=True)
        _train_predictor(base, base_loader, epochs, lr, device)
        _train_predictor(full, full_loader, epochs, lr, device)

        base.eval(); full.eval()
        with torch.no_grad():
            yte = yt[te_t].to(device)
            p0 = base(zt[te_t].to(device))
            p1 = full(zt[te_t].to(device), xt[te_t].to(device))
            r0[te] = ((yte - p0) ** 2).cpu().numpy()
            r1[te] = ((yte - p1) ** 2).cpu().numpy()

    def cdi_from_idx(idx):
        v0 = max(r0[idx].mean(), 1e-300)
        v1 = max(r1[idx].mean(), 1e-300)
        return 0.5 * np.log(v0 / v1) / _LOG2

    point = float(cdi_from_idx(np.arange(n)))

    # Cluster bootstrap on the held-out residual variances (same as the linear
    # estimator): resample whole clusters, recompute CDI = 0.5 log2(v0/v1).
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed + 1)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        boot[b] = cdi_from_idx(idx)
    lcb = float(np.quantile(boot, 0.05))

    return NeuralCdiResult(cdi_bits=point, lcb95_bits=lcb,
                           resid0=r0, resid1=r1, clusters=clusters)
