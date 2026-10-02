"""Large-scale nonlinear Granger causality (lsNGC)."""

from .core import (
    CausalityResult,
    analyze,
    calc_f_stat,
    f_pvalues,
    lsNGC,
    lsngc,
    multivariate_split,
    normalize_0_mean_1_std,
)
from .evaluate import recovery_performance

__version__ = "1.0.0"
__all__ = [
    "CausalityResult", "analyze", "calc_f_stat", "f_pvalues", "lsNGC", "lsngc",
    "multivariate_split", "normalize_0_mean_1_std", "recovery_performance",
]
