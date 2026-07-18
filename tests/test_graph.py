"""The directed-information graph must (a) recover the analytic VAR(1) DI graph,
and (b) drive the MEDIATED edge to zero while keeping the DIRECT edges positive
-- the direct-vs-mediated distinction correlation/PAC cannot make. This is the
whole point of causally-conditioned directed information (Quinn-Kiyavash-Coleman).

Chain structure:  A -> B -> C   (no direct A -> C).
Expect: DI(A->B)>0, DI(B->C)>0, DI(A->C || B) ~ 0, and a naive pairwise
DI(A->C | C_past) > 0 (the spurious edge conditioning removes).

Run: PYTHONPATH=. python tests/test_graph.py
"""

import numpy as np

from neuroforecast.graph import (
    analytic_var1_di_graph,
    directed_information_graph,
)
from neuroforecast.linear import conditional_directed_information


def _simulate_var1(A, T, seed=0):
    rng = np.random.default_rng(seed)
    m = A.shape[0]
    x = np.zeros((T, m))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.standard_normal(m)
    return x


def test_graph_recovers_analytic_and_kills_mediated():
    # A(0) -> B(1) -> C(2); NO direct A -> C.
    A = np.array([
        [0.30, 0.00, 0.00],   # A depends on its own past
        [0.55, 0.30, 0.00],   # B <- A  (direct)
        [0.00, 0.55, 0.30],   # C <- B  (direct);  C <- A only via B
    ])
    names = ["A", "B", "C"]
    truth = analytic_var1_di_graph(A)
    x = _simulate_var1(A, T=40000, seed=0)
    clusters = np.repeat(np.arange(40), len(x) // 40 + 1)[:len(x) - 1]
    g = directed_information_graph(x, names, clusters=clusters, n_boot=300, seed=0)

    print("analytic DI graph (bits), row=source col=dest:\n", np.round(truth, 4))
    print("estimated DI graph (bits):\n", np.round(g.cdi, 4))

    # direct edges recovered and certified
    assert g.lcb[0, 1] > 0, "A->B not certified"
    assert g.lcb[1, 2] > 0, "B->C not certified"
    # MEDIATED edge A->C (conditioned on B) ~ 0 and NOT certified
    assert not (g.lcb[0, 2] > 0.01), f"mediated A->C wrongly certified ({g.lcb[0,2]:.4f})"
    # estimator matches analytic on the direct edges
    assert abs(g.cdi[0, 1] - truth[0, 1]) < 0.02
    assert abs(g.cdi[1, 2] - truth[1, 2]) < 0.02

    # the spurious pairwise edge that conditioning removes: A->C WITHOUT conditioning on B
    future = x[1:]
    a_past = x[:-1, 0:1]
    c_past = x[:-1, 2:3]
    pairwise_ac = conditional_directed_information(future[:, 2], a_past, c_past, n_boot=200)
    print(f"pairwise A->C (no conditioning on B): {pairwise_ac.cdi_bits:+.4f} "
          f"[lcb {pairwise_ac.lcb95_bits:+.4f}]  <- spurious, correlation would show this")
    print(f"conditioned A->C || B:               {g.cdi[0,2]:+.4f} "
          f"[lcb {g.lcb[0,2]:+.4f}]  <- ~0, the mediated edge correctly removed")
    # the meaningful claim: the pairwise view CERTIFIES a spurious A->C edge,
    # the causally-conditioned view does NOT -- exactly what correlation/PAC
    # cannot distinguish and directed-information graphs can.
    assert pairwise_ac.lcb95_bits > 0, "expected pairwise to certify the spurious edge"
    assert g.lcb[0, 2] <= 0, "conditioning failed to remove the mediated edge"


if __name__ == "__main__":
    test_graph_recovers_analytic_and_kills_mediated()
    print("\nDirected-information graph validation passed.")
