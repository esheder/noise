from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

SystemGen = Callable[[float, float, float, "Parameters"], Tuple[float, float]]


# noinspection NonAsciiCharacters
@dataclass(frozen=True, repr=True)
class Parameters:
    """Data object for the underlying problem we're solving.
    Parameters
    ----------
    α - Rate of flux change in the core. [1/sec]
    s - Source rate in the core. [1/sec]
    σ1 - Noise standard deviation for the creation and removal of neutrons in the core.
    σ2 - Noise standard deviation for the detection of neutrons in the core.
    λd - Rate of detection of core flux. [1/sec]
    stepper - Stepping mechanism to move the system forward in time.

    """
    α: float
    s: float
    σ1: float
    σ2: float
    λd: float

    @classmethod
    def from_xs(cls, λa: float, λf: float, ν: float, ν2: float, s: float,
                λd: float) -> "Parameters":
        """Create a parameter object from more trivial cross section data.

        Parameters
        ----------
        λa - Absorption rate in the core. [1/sec]
        λf - Fission rate in the core. [1/sec]
        ν - Mean number of neutrons born in a fission event in the core. [1]
        ν2 - Second moment of the fission multiplicity distribution.
        s - Source rate in the core. [1/sec]
        λd - Rate of detection of core flux. [1/sec]

        Returns
        -------
        A parameter object with the data required for the formalism in the paper.

        """
        α = ν*λf - λa - λd
        σ1 = np.sqrt((s/α) * (λa + λf*(1. - 2.*ν + ν2)) + s)
        σ2 = np.sqrt(λd * s / α)

        return cls(α, s, σ1, σ2, λd)

    @classmethod
    def from_dubi(cls, ρ: float, Λ: float, s: float, ν: float,
                  ν2: float, pd: float) -> "Parameters":
        assert ρ < 0.
        k = 1. / (1. - ρ)
        α = ρ / Λ
        lifetime = Λ * k
        pf = k / ν
        pa = 1. - pd - pf
        λf, λd, λa = [p / lifetime for p in [pf, pd, pa]]
        σ1 = np.sqrt(-(s / α) * (λa + λf * (1. - (2. * ν) + ν2)) + s)
        σ2 = np.sqrt(-λd * s / α)
        return cls(α, s, σ1, σ2, λd)

    @property
    def equilibrium_flux(self) -> float:
        return abs(self.s / self.α)
