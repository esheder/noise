from typing import Tuple

import numpy as np
import pytest
import matplotlib.pyplot as plt

from noise.analog import spread_sources, AnalogParameters, detections, feynman_y
from noise.analytic import fit_to_feynman_y, feynman_y_model


def test_sources():
    rnd = np.random.Generator(np.random.SFC64(48))
    t = 10.
    s = 1e6
    sources = spread_sources(t, s, rnd)
    assert len(sources) == pytest.approx(s*t, rel=1e-3)
    assert max(sources) <= t
    assert min(sources) >= 0.
    assert max(sources) > 0.99*t


@pytest.fixture
def example_par() -> AnalogParameters:
    multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])
    assert multiplicity.sum() == pytest.approx(1., rel=1e-6)
    _mul = multiplicity * np.arange(len(multiplicity))
    _mul2 = multiplicity * (np.arange(len(multiplicity)) ** 2)
    assert _mul.sum() == pytest.approx(2.42), _mul
    k = 0.999
    pf = k / 2.42
    return AnalogParameters(5e-5, 1e6, pf, 1e-3, multiplicity, 600.)


@pytest.fixture
def simulation(example_par: AnalogParameters) -> Tuple[np.array, np.array]:
    assert example_par.k < 1.
    rnd = np.random.Generator(np.random.SFC64(48))
    sources = spread_sources(example_par.tmax, example_par.s, rnd)
    return sources, detections(sources, example_par, rnd)


def test_detections(example_par, simulation):
    sources, signal = simulation
    assert len(signal) == pytest.approx(len(sources)*example_par.pd / (1. - example_par.k), rel=1e-1)


@pytest.fixture
def fy(example_par, simulation) -> Tuple[np.array, np.array, np.array]:
    sources, signal = simulation
    return feynman_y(signal, example_par.tmax)


def test_fy(example_par, fy):
    ts, ys, _ = fy
    mask = (ts <= 1e-1)
    tfit, cfit, = ts[mask], ys[mask]
    popt, pcov = fit_to_feynman_y(tfit, cfit, None)
    assert popt[1] == pytest.approx(-example_par.α, rel=1e-1)


def test_plot_fy(fy):
    ts, ys, _ = fy
    mask = (ts <= 1e-1)
    tfit, cfit, = ts[mask], ys[mask]
    plt.semilogx(ts, ys, '.b')
    try:
        popt, pcov = fit_to_feynman_y(tfit, cfit)
        fitted = feynman_y_model(ts, *popt)
        plt.semilogx(ts, fitted, '-r')
    except RuntimeError:
        pass
    plt.grid()
    plt.show()
