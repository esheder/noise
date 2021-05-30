"""Calculate answers to an SDE of the form:

dφ = -αφdt + Sdt + σ1dW1 - σ2dW2
dψ = λφdt + σ2dW2

"""
from functools import lru_cache
from typing import Tuple
from itertools import takewhile

import numpy as np
from numba import njit

from .parameters import Parameters

rand_gen = np.random.Generator(np.random.SFC64(48))


@njit(nogil=True)
def _hot_loop(x: np.ndarray, dt: np.ndarray, dx: np.ndarray, α: float) -> None:
    for i in range(1, len(x)):
        x[i] = x[i-1] + α*x[i-1]*dt[i-1] + dx[i-1]


def signal_make(ts: np.ndarray, par: Parameters, *, rand_gen: np.random.Generator) -> np.ndarray:
    dt = np.diff(ts)
    rands = rand_gen.normal(0., np.sqrt(dt), (2, len(dt)))
    dnoise = par.σ2 * rands[1, :]
    dx = par.s*dt + par.σ1*rands[0, :] - dnoise
    x = par.equilibrium_flux * np.ones_like(ts, dtype=np.float64)
    _hot_loop(x, dt, dx, par.α)
    y = np.zeros_like(ts, dtype=np.float64)
    y[1:] = par.λd*x[:-1]*dt + dnoise
    return y


def _cut_to_bins(signal: np.ndarray, ts: np.ndarray, skip: int) -> Tuple[float, float, float]:
    skipsignal = np.diff(signal.cumsum()[::skip])
    fy = (skipsignal.var(ddof=1) / skipsignal.mean()) - 1.
    sigma = 1. / np.sqrt(len(skipsignal))
    return ts[skip], fy, sigma


@lru_cache
def feynman_y_by_signal(par: Parameters, t: float,
                        rand_gen: np.random.Generator = rand_gen) -> Tuple[np.array, np.array, np.array]:
    dt = min(0.01/abs(par.α), 1e-3)
    n = int(t/dt) + 1
    ts = np.linspace(0., t, n)
    signal = signal_make(ts, par, rand_gen=rand_gen)
    _times = int(np.log2(n))
    max_interval = min(1., t/8)
    resolutions = takewhile(lambda x: dt*x <= max_interval, map(lambda x: 2**x, range(2*_times)))
    triplets = (_cut_to_bins(signal, ts, skip) for skip in resolutions)
    dts, fy, sigma = list(zip(*triplets))
    return np.array(dts, dtype=np.float64), np.array(fy, dtype=np.float64), np.array(sigma, dtype=np.float64)



