"""Certified directed-information graph over multiple simultaneous signals.

The stomach-brain sleep study (Rao et al. 2025) states its own gap: the analyses
are *correlational* (cross-correlation, phase-amplitude coupling), "precluding
inference about directionality," and names **directed information** [Quinn,
Kiyavash, Coleman 2015, "Directed information graphs"] as the fix. This module is
that fix, made finite-sample-certified and multi-organ.

The estimand is the **causally-conditioned directed information** of Quinn-
Kiyavash-Coleman: for signals {X_1, ..., X_m}, the directed edge i -> j is

    DI(i -> j || rest) = I( X_j(t) ; X_i(past) | X_j(past), X_{k != i,j}(past) ).

Conditioning on *all other signals' pasts* is what separates a DIRECT edge from a
mediated one: if X_i influences X_j only through X_k, then conditioning on X_k's
past drives DI(i -> j || rest) to zero, while a pairwise correlation or PAC (and
even a pairwise transfer entropy) would still light up. That distinction is
exactly what a correlational analysis cannot make and what the lab asked for.

Each edge is estimated with the cross-fitted plug-in from `linear.py` (or the
neural estimator), and certified with a subject-cluster bootstrap lower bound.
The result is a directed graph whose edges are certified in bits.

Validated against the analytic directed-information graph of a linear-Gaussian
VAR(1) system (see `test_graph.py`), where the true graph is the support of the
transition matrix and mediated paths are provably zero after conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroforecast.linear import analytic_cdi_gaussian, conditional_directed_information


@dataclass
class DiGraphResult:
    names: list[str]
    cdi: np.ndarray          # (m, m) point estimates in bits; cdi[i, j] = DI(i -> j || rest)
    lcb: np.ndarray          # (m, m) 95% lower bounds
    floor: float             # certified detection floor (bits) from calibration, if provided

    def certified_edges(self) -> list[tuple[str, str, float]]:
        """Edges whose lower bound clears the detection floor."""
        out = []
        m = len(self.names)
        for i in range(m):
            for j in range(m):
                if i != j and self.lcb[i, j] > max(0.0, self.floor):
                    out.append((self.names[i], self.names[j], float(self.cdi[i, j])))
        return sorted(out, key=lambda e: -e[2])


def _lagged_design(series: np.ndarray, lag: int):
    """From (T, m) multivariate series build future X(t) and past window X(t-1..t-lag).
    Returns future (n, m) and past (n, m, lag)."""
    T, m = series.shape
    n = T - lag
    future = series[lag:]                                  # (n, m)
    past = np.stack([series[lag - k - 1: T - k - 1] for k in range(lag)], axis=-1)  # (n, m, lag)
    return future, past


def directed_information_graph(
    series: np.ndarray,
    names: list[str],
    *,
    lag: int = 1,
    clusters: np.ndarray | None = None,
    floor: float = 0.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> DiGraphResult:
    """Estimate the certified causally-conditioned directed-information graph.

    Args:
        series:   (T, m) simultaneous signals (columns = signals).
        names:    length-m signal names (e.g. ["EEG_sigma","EGG","EKG_HRV","EMG"]).
        lag:      number of past lags used as history.
        clusters: (T-lag,) cluster ids (subject) for the bootstrap.
        floor:    detection floor in bits (from calibration) for edge certification.
    """
    future, past = _lagged_design(series, lag)
    n, m, _ = past.shape
    past_flat = past.reshape(n, m, lag)
    cdi = np.zeros((m, m))
    lcb = np.zeros((m, m))
    for j in range(m):                       # destination
        y = future[:, j]
        for i in range(m):                   # source
            if i == j:
                continue
            others = [k for k in range(m) if k not in (i, j)]
            x_src = past_flat[:, i, :]                       # (n, lag)
            z = np.hstack([past_flat[:, j, :]] +            # dst own past
                          [past_flat[:, k, :] for k in others])  # all other pasts
            res = conditional_directed_information(
                y, x_src, z, clusters=clusters, n_boot=n_boot, seed=seed)
            cdi[i, j] = res.cdi_bits
            lcb[i, j] = res.lcb95_bits
    return DiGraphResult(names=list(names), cdi=cdi, lcb=lcb, floor=float(floor))


# --------------------------------------------------------------------------- #
# Analytic ground truth for a linear-Gaussian VAR(1): the true DI graph.
# --------------------------------------------------------------------------- #
def analytic_var1_di_graph(A: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """Closed-form causally-conditioned DI graph for x(t) = A x(t-1) + eps, eps~N(0,Q).

    Returns (m, m) matrix of DI(i -> j || rest) in bits. Nonzero exactly on the
    support of A (direct edges); mediated paths are zero after conditioning.
    """
    from scipy.linalg import solve_discrete_lyapunov
    m = A.shape[0]
    if Q is None:
        Q = np.eye(m)
    Sigma = solve_discrete_lyapunov(A, Q)               # stationary cov of x(t)
    # joint covariance of [x(t); x(t-1)] : [[Sigma, A Sigma], [Sigma A^T, Sigma]]
    top = np.hstack([Sigma, A @ Sigma])
    bot = np.hstack([Sigma @ A.T, Sigma])
    cov = np.vstack([top, bot])                          # (2m, 2m); block0=x(t), block1=x(t-1)
    out = np.zeros((m, m))
    for j in range(m):
        iy = j                                           # x_j(t)
        for i in range(m):
            if i == j:
                continue
            ix = [m + i]                                 # x_i(t-1)
            iz = [m + j] + [m + k for k in range(m) if k not in (i, j)]  # other pasts
            out[i, j] = analytic_cdi_gaussian(cov, iy=iy, ix=ix, iz=iz)
    return out
