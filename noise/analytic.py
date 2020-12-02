"""Tools for the analytic formula of Feynman-Y

"""

from scipy.optimize import curve_fit
import numpy as np


def feynman_y_model(t: np.ndarray, y_inf: float, α: float) -> np.ndarray:
    return y_inf*(1. - (1. - np.exp(-α*t))/(α*t))


def fit_to_feynman_y(ts, curve, sigma=None):
    return curve_fit(feynman_y_model, ts, curve, np.array([1., 1.]),
                     sigma=sigma,
                     check_finite=False)
