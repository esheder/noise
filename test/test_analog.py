import numpy as np
import pytest

from noise.analog import spread_sources, AnalogParameters, signal_make
from noise.analytic import fit_to_feynman_y, feynman_y_model
from noise.sdesolve import feynman_y_by_signal
import matplotlib.pyplot as plt


def test_sources():
    rnd = np.random.Generator(np.random.SFC64(48))
    t = 10.
    s = 1e6
    sources = spread_sources(t, s, rnd)
    assert len(sources) == pytest.approx(s*t, rel=1e-3)
    assert max(sources) <= t
    assert min(sources) >= 0.
    assert max(sources) > 0.99*t


def test_plot_signal():
    t = 60.
    multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])
    par = AnalogParameters.from_dubi(-500e-5, 5e-5, 1e4, multiplicity, 1e-4)
    ts, curve, _ = feynman_y_by_signal(par, t, signal_generator=signal_make)
    popt, pcov = fit_to_feynman_y(ts, curve, None)
    fitted = feynman_y_model(ts, *popt)
    plt.semilogx(ts, curve, '.b')
    plt.semilogx(ts, fitted, '-r')
    plt.grid()
    plt.show()
