"""Params dataclass.
"""

from dataclasses import dataclass


@dataclass
class Params:
    # sprocess
    substr_n = "1e18"
    """Substrate N conc."""
    epi_thick = "10"
    """Epi thickness in um."""
    epi_p_high = "1e18"
    """Epi P high conc."""
    epi_p_low = "5e15"
    """Epi P low conc."""

    # IdVg
    # TODO the flag at the top
    sdev_params = "sdevice.par"
    idvg_vd = "0.1"
    idvg_vg = "5"
