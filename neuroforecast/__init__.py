"""neuroforecast: certified conditional directed information for the question
'does an added channel improve the forecast of a future neural state, beyond the
target's own past -- and is that improvement real or underpowered?'

Public API:
    linear.conditional_directed_information   -- cross-fitted linear-Gaussian CDI
    linear.analytic_cdi_gaussian              -- closed-form ground truth
    neural.neural_conditional_directed_information -- TCN neural CDI (nonlinear)
    calibration.detection_floor               -- certified detection floor
"""

from neuroforecast.graph import (
    DiGraphResult,
    analytic_var1_di_graph,
    directed_information_graph,
)
from neuroforecast.linear import (
    CdiResult,
    analytic_cdi_gaussian,
    conditional_directed_information,
)

__all__ = [
    "conditional_directed_information",
    "analytic_cdi_gaussian",
    "CdiResult",
    "directed_information_graph",
    "analytic_var1_di_graph",
    "DiGraphResult",
]
