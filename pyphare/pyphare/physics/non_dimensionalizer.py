"""Ion-scale (Alfvenic) non-dimensionalization helper.

Builds a table of reference values from two physical inputs:

  - ``B0`` : reference magnetic field magnitude [T]
  - ``n0`` : reference number density [m^-3]

All other reference quantities derive from ``B0`` and ``n0`` using proton
mass (``m_p``) and vacuum permeability (``mu_0``):

  - Alfven speed         v0     = B0 / sqrt(m_p * mu_0 * n0)
  - ion cyclotron freq   omega0 = e * B0 / m_p
  - ion inertial length  delta0 = v0 / omega0

Use ``NonDimensionalizer(B0, n0)(qty, value)`` to convert a dimensional
``value`` of kind ``qty`` to its non-dimensional counterpart.
The mapping interface (``__getitem__``, iteration, ``len``) exposes the
reference values themselves.
"""

import scipy.constants as sct
from math import sqrt
from enum import Enum, auto
from collections.abc import Mapping, Iterator
from typing import override


class Quantity(Enum):
    """Physical quantity kinds supported by :class:`NonDimensionalizer`."""

    CURRENT = auto()
    DENSITY = auto()
    ELECTRIC_FIELD = auto()
    FREQUENCY = auto()
    HYPER_RESISTIVITY = auto()
    LENGTH = auto()
    MAGNETIC_DIPOLE_2D = auto()
    MAGNETIC_FIELD = auto()
    MASS = auto()
    MASS_DENSITY = auto()
    MOMENTUM_DENSITY = auto()
    PRESSURE = auto()
    RESISTIVITY = auto()
    TIME = auto()
    VELOCITY = auto()


class NonDimensionalizer(Mapping[Quantity, float]):
    """Reference-value table + dimensional-to-nondimensional conversion.

    Parameters
    ----------
    B0 : float
        Reference magnetic field magnitude [T].
    n0 : float
        Reference number density [m^-3].

    Examples
    --------
    >>> nd = NonDimensionalizer(B0=1e-9, n0=5e6)
    >>> nd[Quantity.VELOCITY]      # Alfven speed [m/s]
    >>> nd(Quantity.TIME, 60.0)    # 60 s in units of 1/omega_ci
    """

    def __init__(self, B0: float, n0: float) -> None:
        v0 = B0 / sqrt(sct.m_p * sct.mu_0 * n0)
        omega0 = sct.e * B0 / sct.m_p
        delta0 = v0 / omega0
        self._ref: dict[Quantity, float] = {
            Quantity.CURRENT: B0 / (delta0 * sct.mu_0),
            Quantity.DENSITY: n0,
            Quantity.ELECTRIC_FIELD: v0 * B0,
            Quantity.FREQUENCY: omega0,
            Quantity.HYPER_RESISTIVITY: sct.mu_0 * v0 * delta0**3,
            Quantity.LENGTH: delta0,
            Quantity.MAGNETIC_DIPOLE_2D: B0 * delta0 ** 2 / sct.mu_0,  
            Quantity.MAGNETIC_FIELD: B0,
            Quantity.MASS: sct.m_p,
            Quantity.MASS_DENSITY: sct.m_p * n0,
            Quantity.MOMENTUM_DENSITY: sct.m_p * n0 * v0,
            Quantity.PRESSURE: B0**2 / sct.mu_0,
            Quantity.RESISTIVITY: sct.mu_0 * v0 * delta0,
            Quantity.TIME: 1.0 / omega0,
            Quantity.VELOCITY: v0,
        }

    @override
    def __getitem__(self, qty: Quantity) -> float:
        """Return the reference value associated with ``qty``."""
        return self._ref[qty]

    @override
    def __iter__(self) -> Iterator[Quantity]:
        return iter(self._ref)

    @override
    def __len__(self) -> int:
        return len(self._ref)

    def __call__(self, qty: Quantity, value: float) -> float:
        """Return ``value`` (in SI units) divided by the reference of ``qty``."""
        return value / self._ref[qty]
