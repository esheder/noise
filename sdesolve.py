"""Calculate answers to an SDE of the form:

dφ = -σaφdt + Sdt + νσfφdt -λφdW1
dψ = λφdW1 - ξψdW2 - Ldt
dD = ξψdW2

"""
from functools import lru_cache
from typing import Tuple, Sequence, Iterable

import numpy as np
import matplotlib.pyplot as plt
import pytest
from hypothesis import given, settings, Phase
import hypothesis.strategies as st

from .parameters import Parameters
from .analytic import fit_to_feynman_y, feynman_y_model


rand_gen = np.random.Generator(np.random.SFC64(48))


def _grab_two_random(dt: float) -> np.array:
    std = np.sqrt(dt)
    return rand_gen.normal(0., std, (2,))


def system_step(x0: float, y0: float, dt: float, par: Parameters) -> Tuple[float, float]:
    """Perform one step forward Euler step of the system.

    Parameters
    ----------
    x0 - Core flux at beginning of step.
    y0 - Detector flux at beginning of step
    dt - Step size, in seconds
    par - System parameters.

    Returns
    -------
    The core flux and detector fluxes at end of step

    """
    rands = _grab_two_random(dt)
    x = x0 + par.α*x0*dt + par.s*dt + par.σ1*rands[0] - par.σ2*rands[1]
    y = y0 + par.λd*x0*dt + par.σ2*rands[1]
    return x, y


def gen_count(t: np.array, par: Parameters) -> Iterable[Tuple[float, float]]:
    y0, y1 = par.equilibrium_flux, 0.
    yield y0, y1
    for dt in np.diff(t):
        y0, y1 = par.stepper(y0, y1, dt, par)
        yield y0, y1


def feynman_y_by_signal(t: float, par: Parameters) -> Iterable[Tuple[float, float]]:
    dt = min(0.01/abs(par.α), 1e-3)
    n = int(t/dt) + 1
    ts = np.linspace(0., t, n)
    gen = (v[1] for v in gen_count(ts, par))
    signal = np.fromiter(gen, np.float64, n)
    times = int(np.log2(n))
    for stepsize in [2**i for i in range(times-4)]:
        skipsignal = signal[::stepsize]
        skipsignal = np.diff(skipsignal)
        fy = (skipsignal.var() / skipsignal.mean()) - 1.
        yield ts[stepsize], fy


@lru_cache
def feynman_curve_by_signal(par: Parameters, t: float) -> Tuple[np.array, np.array]:
    pairs = list(feynman_y_by_signal(t, par))
    ts, fy = list(zip(*pairs))
    return np.array(ts, dtype=np.float64), np.array(fy, dtype=np.float64)


times = st.floats(min_value=1., max_value=60.)
reactivities = st.floats(min_value=-5000e-5, max_value=-10e-5)
sources = st.floats(min_value=1e4, max_value=1e8)
detections = st.floats(min_value=1e-5, max_value=1e-3)


@settings(max_examples=5,
          phases=[Phase.explicit, Phase.reuse, Phase.generate],
          deadline=None)
@given(t=times, r=reactivities, s=sources, d=detections)
def test_fitting_signal(t: float, r: float, s: float, d: float):
    par = Parameters.from_dubi(r, 5e-5, s, 2.42, 36., d,
                               stepper=system_step)
    ts, curve = feynman_curve_by_signal(par, t)
    popt, pcov = fit_to_feynman_y(ts, curve)
    print('')
    print(par)
    assert popt[1] == pytest.approx(-par.α, rel=1e-1)


def _test_plot_signal():
    t = 60.
    par = Parameters.from_dubi(-500e-5, 5e-5, 1e6, 2.42, 36., 1e-4,
                               stepper=system_step)
    ts, curve = feynman_curve_by_signal(par, t)
    popt, pcov = fit_to_feynman_y(ts, curve)
    fitted = feynman_y_model(ts, *popt)
    plt.semilogx(ts, curve, '.b')
    plt.semilogx(ts, fitted, '-r')
    plt.show()


