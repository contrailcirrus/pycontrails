"""Fuel data support."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Fuel:
    """Base class for the physical parameters of the fuel."""

    #: Fuel Name
    fuel_name: str

    #: Lower calorific value (LCV) of fuel, :math:`[J \ kg_{fuel}^{-1}]`
    q_fuel: float

    #: Percentage of hydrogen mass content in the fuel
    hydrogen_content: float

    #: CO2 emissions index for fuel, :math:`[kg_{CO_{2}} \ kg_{fuel}^{-1}]`
    ei_co2: float

    #: Water vapour emissions index for fuel, :math:`[kg_{H_{2}O} \ kg_{fuel}^{-1}]`
    ei_h2o: float

    #: Sulphur oxide, SO2-S gas, emissions index for fuel, :math:`[kg_{SO_{2}} \ kg_{fuel}^{-1}]`
    ei_so2: float

    #: Sulphates, S(VI)-S particle, emissions index for fuel, :math:`[kg_{S} \ kg_{fuel}^{-1}]`
    ei_sulphates: float

    #: Organic carbon emissions index for fuel, :math:`[kg_{OC} \ kg_{fuel}^{-1}]`
    ei_oc: float


@dataclasses.dataclass
class JetA(Fuel):
    """Jet A-1 Fuel.

    References
    ----------
    - :cite:`celikel2001forecasting`
    - :cite:`leeContributionGlobalAviation2021`
    - :cite:`stettlerAirQualityPublic2011`
    - :cite:`wilkersonAnalysisEmissionData2010`
    """

    fuel_name: str = "Jet A-1"
    q_fuel: float = 43.13e6
    hydrogen_content: float = 13.8
    ei_co2: float = 3.159
    ei_h2o: float = 1.23

    #: Sulphur oxide, SO2-S gas, emissions index for fuel, :math:`[kg_{SO_{2}} \ kg_{fuel}^{-1}]`
    #: - The EI SO2 is proportional to the fuel sulphur content
    #: - Celikel (2001): EI_SO2 = 0.84 g/kg-fuel for 450 ppm fuel
    #: - Lee et al. (2021): EI_SO2 = 1.2 g/kg-fuel for 600 ppm fuel
    ei_so2: float = 0.0012

    #: Sulphates, S(VI)-S particle, emissions index for fuel, :math:`[kg_{S} \ kg_{fuel}^{-1}]`
    #: - The SOx-S is partitioned into 98% SO2-S gas and 2% S(VI)-S particle
    #: - References: Wilkerson et al. (2010) & Stettler et al. (2011)
    ei_sulphates: float = ei_so2 / 0.98 * 0.02

    #: Organic carbon emissions index for fuel, :math:`[kg_{OC} \ kg_{fuel}^{-1}]`
    #: - High uncertainty
    #: - Wilkerson et al. (2010): EI_OC = 15 mg/kg-fuel
    #: - Stettler et al. (2011): EI_OC = 20 [1, 40] mg/kg-fuel
    ei_oc: float = 20 * 1e-6


class SAFBlend(Fuel):
    """Jet A-1 / Sustainable Aviation Fuel Blend.

    SAF only changes the CO2 lifecycle emissions, not the CO2 emissions emitted at the
    aircraft exhaust. We assume that the EI OC stays the same as Jet A-1 fuel due to lack
    of data.

    Parameters
    ----------
    pct_blend : float
        Sustainable aviation fuel percentage blend ratio by volume, %. Expected
        to be in the interval [0, 100].

    References
    ----------
    - :cite:`teohTargetedUseSustainable2022`
    - :cite:`schrippAircraftEngineParticulate2022`
    """

    def __init__(self, pct_blend: float) -> None:
        if pct_blend < 0.0 or pct_blend > 100.0:
            raise ValueError("pct_blend only accepts a value of between 0 and 100.")

        self.pct_blend = pct_blend

        self.fuel_name = "Jet A-1 / Sustainable Aviation Fuel Blend"

        # We take the default values for Jet-A and modify them for a custom blend
        base_fuel = JetA()
        self.q_fuel = base_fuel.q_fuel + (10700 * self.pct_blend)
        self.hydrogen_content = base_fuel.hydrogen_content + 0.015 * self.pct_blend
        self.ei_co2 = base_fuel.ei_co2
        self.ei_h2o = base_fuel.ei_h2o * (self.hydrogen_content / base_fuel.hydrogen_content)
        self.ei_so2 = base_fuel.ei_so2 * (1 - self.pct_blend / 100)
        self.ei_sulphates = self.ei_so2 / 0.98 * 0.02
        self.ei_oc = base_fuel.ei_oc


@dataclasses.dataclass
class HydrogenFuel(Fuel):
    """Hydrogen Fuel.

    References
    ----------
    - :cite:`khanEmissionsWaterVapour2022`
    """

    fuel_name: str = "Hydrogen"
    q_fuel: float = 122.8e6
    hydrogen_content: float = np.nan
    ei_co2: float = 0.0
    ei_h2o: float = 9.21
    ei_so2: float = 0.0
    ei_sulphates: float = 0.0
    ei_oc: float = 0.0
