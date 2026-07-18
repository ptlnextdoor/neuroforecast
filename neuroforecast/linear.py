"""Conditional directed information (CDI) with a certified detection floor.

The question this instrument answers, in one line:

    Does the causal history of channel X add information about the future of
    channel Y, *beyond* a nuisance baseline Z (which includes Y's own past) --
    and is that added information statistically real or merely underpowered?

This is transfer entropy / causally-conditioned directed information

    CDI = I(Y_future ; X_past | Z),   Z = (Y_past, nuisance),

reported in bits per sample. When Z is the semi-Markov / autoregressive baseline,
CDI is exactly the "residual, transportable, causal information" that a forecast
of Y can draw from X and nothing else.

Two design commitments make the number trustworthy rather than optimistic:

1. Cross-fitted plug-in estimation. We estimate CDI as the held-out predictive
   log-loss improvement of a model given (X, Z) over a model given Z alone:

       CDI_hat = mean_heldout[ log p(y | x, z) - log p(y | z) ] / ln 2.

   Because both terms are evaluated out-of-fold, the estimator does not reward a
   model for overfitting X; a channel that carries no information contributes ~0,
   not a positive bias. For a fixed model class this is a *lower bound* on the
   true CDI (the model cannot extract more than exists) -- the honest direction
   to err in.

2. A detection-power calibration. Estimator noise means a small true CDI can
   look null. We inject a synthetic channel carrying a *known* amount of directed
   information at graded strengths and record the strength at which the
   cluster-bootstrap lower bound first clears zero. That strength is the
   instrument's detection floor: below it, a null is uninformative; above it, a
   null is a real negative. This is what turns "we found nothing" into "we could
   have found this much, and there is none."

Linear-Gaussian channels have a closed-form CDI, so the estimator is validated
against analytic ground truth (see `analytic_cdi_gaussian` and `test_cdi.py`).

No framework, ~200 lines, numpy + scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

_LOG2 = np.log(2.0)


def _as2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def _held_out_residual_var(target: np.ndarray, feats: np.ndarray,
                           n_splits: int, alpha: float, seed: int) -> np.ndarray:
    """Out-of-fold residuals of ridge(target ~ feats); return squared residuals
    aligned to the original row order. Cross-fitting removes optimism: an
    uninformative feature block yields residuals no smaller than the baseline's.
    """
    n = len(target)
    resid = np.empty(n, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in kf.split(feats):
        mu = feats[tr].mean(0)
        sd = feats[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)  # constant column -> leave as-is, don't blow up
        ftr = (feats[tr] - mu) / sd
        fte = (feats[te] - mu) / sd
        model = Ridge(alpha=alpha)
        # NumPy 2.x raises a spurious FP flag inside BLAS matmul (X.T@X) during
        # the Cholesky solve; the solve itself is exact. Silence the false flag.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model.fit(ftr, target[tr])
            resid[te] = target[te] - model.predict(fte)
    return resid ** 2


@dataclass
class CdiResult:
    cdi_bits: float               # point estimate (bits / sample)
    lcb95_bits: float             # cluster-bootstrap 95% lower bound
    n: int
    n_clusters: int

    @property
    def is_certified_positive(self) -> bool:
        return self.lcb95_bits > 0.0


def conditional_directed_information(
    y_future: np.ndarray,
    x_past: np.ndarray,
    z_baseline: np.ndarray,
    *,
    clusters: np.ndarray | None = None,
    n_splits: int = 5,
    alpha: float = 1e-3,
    n_boot: int = 2000,
    seed: int = 0,
) -> CdiResult:
    """Estimate CDI = I(y_future ; x_past | z_baseline) in bits/sample.

    Args:
        y_future:   (n,) scalar future target of channel Y.
        x_past:     (n, dx) causal history features of the candidate channel X.
        z_baseline: (n, dz) nuisance/autoregressive baseline (must include Y's own
                    past for CDI to be a *directed* / transfer-entropy quantity).
        clusters:   (n,) cluster ids (e.g. subject) for the bootstrap; if None,
                    each sample is its own cluster (i.i.d. bootstrap).
        n_splits:   cross-fitting folds.
        alpha:      ridge penalty (small -> ~OLS, recovers linear-Gaussian CDI).
        n_boot:     cluster-bootstrap replicates for the lower bound.

    Returns:
        CdiResult with point estimate and one-sided 95% lower bound.
    """
    y = np.asarray(y_future, dtype=np.float64).ravel()
    X = _as2d(x_past)
    Z = _as2d(z_baseline)
    n = len(y)
    if not (len(X) == len(Z) == n):
        raise ValueError("y_future, x_past, z_baseline must share the same length")
    if clusters is None:
        clusters = np.arange(n)
    clusters = np.asarray(clusters)

    r0 = _held_out_residual_var(y, Z, n_splits, alpha, seed)              # given Z
    r1 = _held_out_residual_var(y, np.hstack([X, Z]), n_splits, alpha, seed)  # given X,Z

    def cdi_from_idx(idx: np.ndarray) -> float:
        v0 = r0[idx].mean()
        v1 = r1[idx].mean()
        # Gaussian predictive log-loss improvement = 0.5 log(v0/v1) nats/sample.
        return 0.5 * np.log(max(v0, 1e-300) / max(v1, 1e-300)) / _LOG2

    point = cdi_from_idx(np.arange(n))

    # Cluster (subject) bootstrap: resample whole clusters, recompute CDI.
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in picks])
        boot[b] = cdi_from_idx(idx)
    lcb = float(np.quantile(boot, 0.05))

    return CdiResult(cdi_bits=float(point), lcb95_bits=lcb,
                     n=n, n_clusters=len(uniq))


def analytic_cdi_gaussian(cov: np.ndarray, iy: int, ix: list[int], iz: list[int]) -> float:
    """Closed-form I(Y; X | Z) in bits for a jointly-Gaussian vector with
    covariance `cov`. Y is a single index `iy`; X and Z are index lists.

    I(Y;X|Z) = 0.5 log2( var(Y|Z) / var(Y|X,Z) ),
    with var(Y|W) = Sigma_YY - Sigma_YW Sigma_WW^{-1} Sigma_WY.
    """
    def cond_var(cond: list[int]) -> float:
        syy = cov[iy, iy]
        if not cond:
            return float(syy)
        syw = cov[iy, cond]
        sww = cov[np.ix_(cond, cond)]
        return float(syy - syw @ np.linalg.solve(sww, syw))

    v_z = cond_var(list(iz))
    v_xz = cond_var(list(ix) + list(iz))
    return 0.5 * np.log(v_z / v_xz) / _LOG2
