"""Validation: the estimator recovers the analytic linear-Gaussian CDI, and
reports ~0 (with LCB<=0) when the candidate channel carries no information.

Run: python test_cdi.py
"""

import numpy as np

from neuroforecast.linear import analytic_cdi_gaussian, conditional_directed_information


def _make_gaussian(n, a_xy, b_zy, seed=0):
    """Directed system: X and Z are exogenous; Y_future = a_xy*X + b_zy*Z + noise.
    Returns samples and the joint covariance in order [Y, X, Z].
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y = a_xy * x + b_zy * z + noise
    samples = dict(y=y, x=x, z=z)
    # Analytic covariance of [Y, X, Z] under unit-variance independent x,z,noise.
    var_y = a_xy**2 + b_zy**2 + 1.0
    cov = np.array([
        [var_y, a_xy, b_zy],
        [a_xy, 1.0, 0.0],
        [b_zy, 0.0, 1.0],
    ])
    return samples, cov


def test_recovers_analytic_cdi():
    """Estimated CDI matches the closed-form value across coupling strengths."""
    for a in (0.0, 0.3, 0.6, 1.0, 1.5):
        s, cov = _make_gaussian(20000, a_xy=a, b_zy=0.8, seed=1)
        true = analytic_cdi_gaussian(cov, iy=0, ix=[1], iz=[2])
        est = conditional_directed_information(
            s["y"], s["x"], s["z"], n_boot=200, seed=1)
        err = abs(est.cdi_bits - true)
        print(f"a={a:.2f}  true={true:+.4f}  est={est.cdi_bits:+.4f}  "
              f"lcb={est.lcb95_bits:+.4f}  |err|={err:.4f}")
        assert err < 0.02, f"CDI estimate off by {err:.4f} at a={a}"


def test_null_channel_is_not_positive():
    """A candidate channel with zero true CDI must not certify positive."""
    s, _ = _make_gaussian(20000, a_xy=0.0, b_zy=0.8, seed=2)
    est = conditional_directed_information(s["y"], s["x"], s["z"], n_boot=500, seed=2)
    print(f"null channel: est={est.cdi_bits:+.4f}  lcb={est.lcb95_bits:+.4f}")
    assert not est.is_certified_positive, "null channel wrongly certified positive"


def test_real_channel_is_certified():
    """A channel carrying real directed information must certify positive."""
    s, _ = _make_gaussian(20000, a_xy=0.8, b_zy=0.8, seed=3)
    est = conditional_directed_information(s["y"], s["x"], s["z"], n_boot=500, seed=3)
    print(f"real channel: est={est.cdi_bits:+.4f}  lcb={est.lcb95_bits:+.4f}")
    assert est.is_certified_positive, "real channel failed to certify positive"


if __name__ == "__main__":
    test_recovers_analytic_cdi()
    test_null_channel_is_not_positive()
    test_real_channel_is_certified()
    print("\nAll CDI validation checks passed.")
