from functools import partial

import pytest
from hypothesis import strategies as st, settings, Phase, given
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from noise import Parameters
from noise.analytic import fit_to_feynman_y, feynman_y_model
from noise.sdesolve import feynman_y_by_signal

times = st.floats(min_value=60., max_value=600.)
reactivities = st.floats(min_value=-1000e-5, max_value=-60e-5)
sources = st.floats(min_value=1e7, max_value=1e9)
detections = st.floats(min_value=1e-5, max_value=1e-3)


@settings(max_examples=50,
          phases=[Phase.explicit, Phase.reuse, Phase.generate],
          deadline=None)
@given(t=times, r=reactivities, s=sources, d=detections)
def test_fitting_signal(t: float, r: float, s: float, d: float):
    par = Parameters.from_dubi(r, 5e-5, s, 2.42, 36., d)
    ts, curve, sigma = feynman_y_by_signal(par, t)
    mask = (ts <= 1e-1)
    tfit, cfit, sfit = ts[mask], curve[mask], sigma[mask]
    popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
    try:
        assert popt[1] == pytest.approx(-par.α, rel=1e-1)
    except AssertionError:
        fitted = feynman_y_model(ts, *popt)
        true_alpha_model = partial(feynman_y_model, α=-par.α)
        true_opt, true_cov = curve_fit(true_alpha_model, ts, curve, 1., sigma=sigma, check_finite=False)
        true_alpha = true_alpha_model(ts, *true_opt)
        plt.figure()
        plt.errorbar(ts, curve, yerr=sigma, fmt='.b', label='Data')
        plt.plot(ts, fitted, '-r', label="2-parameter fit")
        plt.plot(ts, true_alpha, '--k', label="Fit with known α")
        plt.grid()
        plt.legend()
        plt.gca().set_xscale('log')
        plt.savefig('failure.png')
        raise


def test_plot_signal():
    t = 60.
    par = Parameters.from_dubi(-500e-5, 5e-5, 1e6, 2.42, 36., 1e-4)
    ts, curve, _ = feynman_y_by_signal(par, t)
    popt, pcov = fit_to_feynman_y(ts, curve, None)
    fitted = feynman_y_model(ts, *popt)
    plt.semilogx(ts, curve, '.b')
    plt.semilogx(ts, fitted, '-r')
    plt.grid()
    plt.show()
