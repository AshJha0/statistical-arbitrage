"""statarb — statistical arbitrage toolkit (cointegration, Kalman, backtest).

Self-contained reference implementation: numpy only.  See API_SPEC.md at
the project root for the pinned numerical contract shared by all language
implementations.
"""
from .adf import (
    CRITICAL_VALUES,
    MIN_NOBS,
    RESPONSE_SURFACE,
    ADFResult,
    adf_test,
    critical_values,
    interpolate_pvalue,
    select_lag,
)
from .backtest import (
    BacktestConfig,
    BacktestResult,
    RollingBlock,
    RollingWalkForwardResult,
    Trade,
    WalkForwardResult,
    backtest_pair,
    compute_metrics,
    config_from_dict,
    rolling_walk_forward,
    walk_forward,
)
from .data_io import load_config, load_csv
from .engle_granger import EGBothResult, EGResult, engle_granger, engle_granger_both
from .johansen import (
    TRACE_CRIT_95,
    JohansenResult,
    cholesky_lower,
    jacobi_eigh,
    johansen_trace,
)
from .kalman import KalmanResult, KalmanStep, kalman_hedge
from .ols import OLSResult, ols
from .signals import RollingZ, half_life, rolling_zscore, spread

__all__ = [
    "ADFResult",
    "BacktestConfig",
    "BacktestResult",
    "CRITICAL_VALUES",
    "EGResult",
    "JohansenResult",
    "KalmanResult",
    "MIN_NOBS",
    "OLSResult",
    "RESPONSE_SURFACE",
    "TRACE_CRIT_95",
    "EGBothResult",
    "KalmanStep",
    "RollingBlock",
    "RollingWalkForwardResult",
    "RollingZ",
    "Trade",
    "WalkForwardResult",
    "adf_test",
    "backtest_pair",
    "cholesky_lower",
    "compute_metrics",
    "config_from_dict",
    "critical_values",
    "engle_granger",
    "engle_granger_both",
    "half_life",
    "interpolate_pvalue",
    "jacobi_eigh",
    "johansen_trace",
    "kalman_hedge",
    "load_config",
    "load_csv",
    "ols",
    "rolling_walk_forward",
    "rolling_zscore",
    "select_lag",
    "spread",
    "walk_forward",
]

__version__ = "1.2.1"
