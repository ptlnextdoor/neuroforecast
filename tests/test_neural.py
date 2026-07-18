"""The neural estimator must (a) recover the analytic CDI on linear-Gaussian
data where the truth is known, and (b) not certify a null channel. This is the
sanity gate: if the neural version can't match the closed form on linear data,
it can't be trusted on nonlinear data where no ground truth exists.

Run: python -m tests.test_neural   (from repo root)
"""

import numpy as np

from neuroforecast.linear import analytic_cdi_gaussian
from neuroforecast.neural import neural_conditional_directed_information


def _linear_system(n, strength, seed):
    """brain_future = 0.6*brain_past + strength*gut_past + 0.5*circadian + noise.
    X_past presented as a length-8 window whose LAST sample carries gut_past
    (so the causal encoder must find the informative timestep)."""
    rng = np.random.default_rng(seed)
    brain_past = rng.standard_normal(n)
    gut_past = rng.standard_normal(n)
    circ = rng.standard_normal(n)
    y = 0.6 * brain_past + strength * gut_past + 0.5 * circ + rng.standard_normal(n)
    # window: 1 channel x 8 samples carrying gut_past across time + small noise
    # (a cleanly learnable signal -- this validates the CDI machinery, not the
    # encoder's needle-in-noise capacity, which is a separate design question).
    win = (gut_past[:, None, None]
           + 0.3 * rng.standard_normal((n, 1, 8))).astype(np.float32)
    Z = np.column_stack([brain_past, circ]).astype(np.float32)
    var_y = 0.6**2 + strength**2 + 0.5**2 + 1.0
    cov = np.array([[var_y, strength, 0.6, 0.5],
                    [strength, 1, 0, 0],
                    [0.6, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    true = analytic_cdi_gaussian(cov, iy=0, ix=[1], iz=[2, 3])
    clusters = rng.integers(0, 20, size=n)
    return y, win, Z, clusters, true


def test_neural_recovers_linear_cdi():
    y, x, z, cl, true = _linear_system(6000, strength=0.8, seed=0)
    res = neural_conditional_directed_information(
        y, x, z, clusters=cl, epochs=120, n_boot=300, seed=0)
    print(f"true={true:+.4f}  neural_est={res.cdi_bits:+.4f}  "
          f"lcb={res.lcb95_bits:+.4f}")
    # neural estimate should be within 0.05 bits of truth and certify positive
    assert abs(res.cdi_bits - true) < 0.05, f"off by {abs(res.cdi_bits-true):.4f}"
    assert res.is_certified_positive


def test_neural_null_not_certified():
    y, x, z, cl, true = _linear_system(6000, strength=0.0, seed=1)
    res = neural_conditional_directed_information(
        y, x, z, clusters=cl, epochs=120, n_boot=300, seed=1)
    print(f"null: true={true:+.4f}  neural_est={res.cdi_bits:+.4f}  "
          f"lcb={res.lcb95_bits:+.4f}")
    assert not res.is_certified_positive, "null channel wrongly certified"


if __name__ == "__main__":
    test_neural_recovers_linear_cdi()
    test_neural_null_not_certified()
    print("\nNeural CDI validation passed.")
