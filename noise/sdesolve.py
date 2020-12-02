"""Calculate answers to an SDE of the form:

dφ = -αφdt + Sdt + σ1dW1 - σ2dW2
dψ = λφdt + σ2dW2

"""
from functools import lru_cache
from typing import Tuple, Iterable
from scipy.stats import kurtosis

import numpy as np

from .parameters import Parameters

rand_gen = np.random.Generator(np.random.SFC64(48))


def signal_make(ts: np.ndarray, par: Parameters, *, randgen: np.random.Generator) -> np.ndarray:
    x0 = par.equilibrium_flux
    dt = np.diff(ts)
    rands = randgen.normal(0., np.sqrt(dt), (2, len(dt)))
    dnoise = par.σ2 * rands[1, :]
    dx = par.s*dt + par.σ1*rands[0, :] - dnoise
    x = x0 * np.ones_like(ts, dtype=np.float64)
    for i in range(1, len(ts)):
        x[i] = x[i-1] + par.α*x[i-1]*dt[i-1] + dx[i-1]
    y = np.zeros_like(ts, dtype=np.float64)
    y[1:] = np.cumsum(par.λd*x[:-1]*dt + dnoise)
    return y


def feynman_y_by_signal(t: float, par: Parameters) -> Iterable[Tuple[float, float, float]]:
    dt = min(0.01/abs(par.α), 1e-3)
    n = int(t/dt) + 1
    ts = np.linspace(0., t, n)
    signal = signal_make(ts, par, randgen=rand_gen)
    _times = int(np.log2(n))
    for stepsize in [2**i for i in range(_times-1)]:
        skipsignal = signal[::stepsize]
        skipsignal = np.diff(skipsignal)
        fy = (skipsignal.var(ddof=1) / skipsignal.mean()) - 1.
        sigma = 1. / np.sqrt(len(skipsignal))
        yield ts[stepsize], fy, sigma


@lru_cache
def feynman_curve_by_signal(par: Parameters, t: float) -> Tuple[np.array, np.array, np.array]:
    triplets = list(feynman_y_by_signal(t, par))
    ts, fy, sigma = list(zip(*triplets))
    return np.array(ts, dtype=np.float64), np.array(fy, dtype=np.float64), np.array(sigma, dtype=np.float64)


