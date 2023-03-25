"""Meteorological, thermophysical, and aircraft constants."""

from __future__ import annotations

# -------
# General
# -------

# NOTE: Use a decimal point for each float-valued constant. This is important for
# converting to numpy arrays.

#: Absolute zero value :math:`[C]`
absolute_zero: float = -273.15

#: Gravitational acceleration :math:`[m \ s^{-2}]`
g: float = 9.80665

#: Radius of Earth :math:`[m]`
radius_earth: float = 6371229.0

#: ISA height of the tropopause :math:`[m]`
#: :cite:`wikipediacontributorsInternationalStandardAtmosphere2023`
h_tropopause: float = 11000.0

# -------------
# Thermodynamic
# -------------

#: Surface pressure, international standard atmosphere :math:`[Pa]`
#: :cite:`wikipediacontributorsInternationalStandardAtmosphere2023`
p_surface: float = 101325.0

#: Isobaric heat capacity of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
c_pd: float = 1004.0  # 1005.7?

#: Isobaric heat capacity of water vapor :math:`[J \ kg^{-1} \ K^{-1}]`
c_pv: float = 1870.0

#: Molecular mass of dry air :math:`[kg \ mol^{-1}]`
M_d: float = 28.9647e-3

#: Molecular mass of water :math:`[kg \ mol^{-1}]`
M_v: float = 18.0153e-3

#: Ratio of heat capacities, TODO: which heat capacities?
gamma: float = 1.4

#:  molar gas constant :math:`[J \ mol^{-1} \ K^{-1}]`
R: float = 8.314462618

#: Gas constant of dry air :math:`[J \ kg^{-1} \ K^{-1}]`
R_d: float = 287.05

#: Gas constant of water vapour :math:`[J \ kg^{-1} \ K^{-1}]`
R_v: float = 461.51

#: Ratio of gas constant for dry air / gas constant for water vapor
epsilon: float = R_d / R_v

#: Density of ice  :math:`[kg \ m^{-3}]`
rho_ice: float = 917.0

#: Adiabatic index air
kappa: float = 1.4

#: Standard atmospheric density at mean sea level (MSL) :math:`[kg \ m^{-3}]`
rho_msl: float = 1.225

#: Standard atmospheric temperature at mean sea level (MSL) :math:`[K]`
T_msl: float = 288.15

#: Speed of sound at mean sea level (MSL) in standard atmosphere :math:`[m \ s^{-1}]`
c_msl: float = 340.294

#: The rate at which the ISA ambient temperature falls with altitude :math:`[K \ m^{-1}]`
#: :cite:`wikipediacontributorsInternationalStandardAtmosphere2023`
T_lapse_rate: float = -0.0065

#: Average incident solar radiation, :math:`[W \ m^{-2}]`
#: This value can range +/- 3% as the earth orbits the sun.
#: From :cite:`UOSRMLSolar` citing :cite:`paltridgeRadiativeProcessesMeteorology1976`
solar_constant: float = 1361.0

# ----
# Fuel
# ----

#: Isobaric heat capacity of combustion products :math:`[J \ kg^{-1} \ K^{-1}]`
c_p_combustion: float = 1250.0

# ------------------
# Optical Properties
# ------------------

#: Real refractive index of ice
mu_ice: float = 1.31

#: Wavelength of visible light (550 nm)
lambda_light: float = 550e-9

#: Ratio between the volume mean radius and the effective radius (Uncertainty +- 0.3)
c_r: float = 0.9

# ------
# Flight
# ------

#: Nominal rate of climb/descent of aircraft [:math:`m \ s^{-1}``]
nominal_rocd: float = 12.7
