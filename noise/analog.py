"""We have a stochastic process. Neutrons are born in the core randomly. Sometimes they die, sometimes they cause
fission and rarely they are observed and die. We want to have a signal of counts within time windows, and we want
to observe the mean and variance of said counts in said windows.

"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence
import logging

import numpy as np


logger = logging.getLogger(__name__)


class Process(IntEnum):
    Detection = 0
    Fission = 1
    Loss = 2


@dataclass
class AnalogParameters:
    """Parameters used in the Analog model.

    Parameters
    ----------
    lifetime - Neutron lifetime, in seconds.
    s - Source rate, in 1/sec.
    pf - Fission probability.
    pd - Detection probability.
    multiplicity - Multiplicity probability vector.
    tmax - Time when the experiment ends.

    """
    lifetime: float
    s: float
    pf: float
    pd: float
    multiplicity: Sequence[float]

    @classmethod
    def from_dubi(cls, ρ: float, Λ: float, s: float, multiplicity: Sequence[float], pd: float):
        k = 1./(1.-ρ)
        lifetime = Λ*k
        nu = (multiplicity * np.arange(len(multiplicity))).sum()
        pf = k/nu
        return cls(lifetime, s, pf, pd, multiplicity)

    @property
    def prob_vector(self) -> tuple[float, ...]:
        return self.pd, self.pf, 1. - self.pd - self.pf

    @property
    def k(self) -> float:
        return float(self.pf*np.sum(np.arange(len(self.multiplicity))*self.multiplicity))

    @property
    def ρ(self) -> float:
        return 1. - (1. / self.k)

    @property
    def α(self) -> float:
        return self.ρ / self.lifetime

    @property
    def pa(self) -> float:
        return 1. - self.pf - self.pd

    def __hash__(self):
        return hash((self.lifetime, self.s, self.pf, self.pd, tuple(self.multiplicity)))


def spread_sources(t: float, s: float,
                   rand_gen: np.random.Generator,
                   shift: float = 0.) -> np.array:
    points = int(s * t)
    dt = rand_gen.exponential(scale=1. / s, size=(points,))
    ts = np.cumsum(dt)
    return ts + shift


def signal_make(ts: np.array,
                par: AnalogParameters,
                *,
                chunks=6_000_000,
                rand_gen: np.random.Generator
                ) -> np.array:
    tmax = ts[-1]
    res = np.zeros_like(ts[:-1])
    shifts, interval = np.linspace(0., tmax, num=max(1, int(tmax*par.s//chunks)), retstep=True, endpoint=False)
    for shift in shifts:
        sources = shift + spread_sources(interval, par.s, rand_gen=rand_gen)
        sources = sources[sources < tmax]
        while len(sources):
            lags = rand_gen.exponential(par.lifetime, size=sources.size)
            tevents = sources + lags
            processes = rand_gen.choice(a=[e.value for e in Process], p=par.prob_vector, size=sources.size)
            detect = processes == Process.Detection.value
            fission = processes == Process.Fission.value
            legal = tevents <= tmax
            res += np.histogram(tevents[detect & legal], bins=ts)[0]
            fission_times = tevents[fission & legal]
            multiples = rand_gen.choice(a=np.arange(len(par.multiplicity)), p=par.multiplicity, size=fission_times.size)
            sources = np.repeat(fission_times, multiples)
    return res
