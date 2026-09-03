"""gwc -- Generalized WhiteBox Challenge.

FLOP-budgeted estimation of per-neuron activation means in random MLPs, across
18 widths x 11 depths x 8 activations x 4 weight distributions. See README.md.
"""

from gwc.activations import NAMES as ACTIVATIONS, apply as activation
from gwc.budget import GROUND_TRUTH_SAMPLES, N_REF, flop_budget, mc_at_budget, mc_flops_per_sample
from gwc.netspec import DEPTHS, WIDTHS, Network, NetType, load_networks
from gwc.sdk import API_VERSION, BaseEstimator, SetupContext
from gwc.weights import STRATEGIES

__version__ = "0.1.0"
__all__ = [
    "ACTIVATIONS", "STRATEGIES", "WIDTHS", "DEPTHS", "N_REF", "GROUND_TRUTH_SAMPLES",
    "Network", "NetType", "BaseEstimator", "SetupContext", "API_VERSION",
    "activation", "flop_budget", "mc_flops_per_sample", "mc_at_budget", "load_networks",
]
