"""Photochemical trajectory model for the Earth's atmosphere."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload
import numpy as np
import datetime
import boxm_for

import xarray as xr

import pycontrails
from pycontrails.core import datalib
from pycontrails.core.flight import Flight
from pycontrails.core.met import MetDataArray, MetDataset, standardize_variables
from chem import ChemDataset
from pycontrails.core.met_var import (
    AirTemperature,
    RelativeHumidity,
    SpecificHumidity,
    AirPressure
)
from pycontrails.core.models import Model, ModelParams
from pycontrails.core.vector import GeoVectorDataset
from pycontrails.datalib import ecmwf
from pycontrails.physics import geo, thermo, units, constants
   
@dataclass
class ChemParams(ModelParams):
    """Default trajectory model parameters."""
    lat_bound: tuple[float, float] | None = None
    lon_bound: tuple[float, float] | None = None
    alt_bound: tuple[float, float] | None = None
    start_date: str = "2021-01-01"
    start_time: str = "00:00:00"
    chem_ts: int = 60 # seconds between chemistry calculations
    disp_ts: int = 300 # seconds between dispersion calculations
    runtime: int = 24 # hours model runtime
    horiz_res: float = 0.25 # degrees
    bgoam: float = 0.7 # background organic aerosol mass
    microgna: float = 0.0 # microgram of nitrate aerosol
    microgsa: float = 0.0 # microgram of sulfate aerosol


class BoxModel(Model):
    """Compute chemical concentrations along a trajectory."""
    name = "boxm"
    long_name = "Photochemical Trajectory Model"
    met_variables = (
        AirTemperature,
        SpecificHumidity,
        RelativeHumidity,
        AirPressure
    )
    default_params = ChemParams

    # Met, chem data is not optional
    met: MetDataset
    chem: ChemDataset
    met_required = True

    timesteps: npt.NDArray[np.datetime64]


    def __init__(
        self,
        met: MetDataset,
        chem: MetDataset,
        params: dict[str, Any] | None = None,
        **params_kwargs: Any,
    ):
        
        # Normalize ECMWF variables
        #met = standardize_variables(met, self.met_variables)

        super().__init__(met, params=params, **params_kwargs)
        
        self.met = met
        self.chem = chem
        #self.met = self.require_source_type(MetDataset)
        #self.chem = self.require_source_type(ChemDataset)

        self.met["air_pressure"] = self.met.broadcast_coords("air_pressure")

        self.timesteps = self.chem.data["time"].values

    # ----------
    # Public API
    # ----------

    def eval(
        self, source: ChemDataset | None = None, **params: Any
    ) -> ChemDataset:
        """Evaluate chem on meteorology grid, subject to flight emissions.
        
        Parameters
        ----------
        source : MetDataset | None, optional
            Input MetDataset
            If None, evaluates at the :attr:`met` grid points.
        **params : Any
            Overwrite model parameters before eval

        Returns
        -------
        MetDataset
            Returns `np.nan` if inte rpolating outside meteorology grid.

        Raises
        ------
        NotImplementedError
            Raises if input ``source`` is not supported.       
        """

        self.update_params(params)
        self.source = source
       
        # Interpolate emi to chem grid
        self.source.data = self.source.data.interp_like(self.chem.data)

        # Assign emissions dataarray to chem dataset
        self.chem["EM"].data = self.source["EM"].data

        # Export all variables to fortran
        self._f2py_export()
        
        # Calculate chemical concentrations and flux rates for each timestep (requires iteration)
        for time_idx, ts in enumerate(self.timesteps):
           
            print(time_idx, ts)
        
            # Calculate mass of organic particulate material and mass of organic aerosol
            self.calc_aerosol(ts)

            self._chemco(ts)

            self._photol(ts)

            if time_idx != 0:
                self._deriv(time_idx)

        self._f2py_import()

    # -------------
    # Model Methods
    # -------------

    def _f2py_export(self):
        met = self.met
        chem = self.chem

        N_A = 6.022e23 # Avogadro's number
        
        # Get air density from pycontrails physics.thermo script
        rho_d = thermo.rho_d(met["air_temperature"].data.values, 
                             met["air_pressure"].data.values)

        # Calculate number density of air (M) to feed into box model calcs
        M = (N_A / constants.M_d) * rho_d * 1e-6 # [molecules / cm^3]

        # Calculate H2O number concentration to feed into box model calcs
        H2O = (met["specific_humidity"].data.values / constants.M_v) * N_A * rho_d * 1e-6 
        # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        O2 = 2.079E-01*M
        N2 = 7.809E-01*M        

        # Clear all variables prior to allocation
        boxm_for.boxm.temp = None
        boxm_for.boxm.pressure = None
        boxm_for.boxm.spec_hum = None
        boxm_for.boxm.m = None
        boxm_for.boxm.h2o = None
        boxm_for.boxm.o2 = None

        boxm_for.boxm.y = None
        #boxm_for.boxm.yp = None
        boxm_for.boxm.rc = None
        boxm_for.boxm.dj = None
        boxm_for.boxm.em = None
        boxm_for.boxm.fl = None

        boxm_for.boxm.j = None
        boxm_for.boxm.soa = None
        boxm_for.boxm.mom = None
        boxm_for.boxm.br01 = None
        boxm_for.boxm.ro2 = None

        # Position and time
        boxm_for.boxm.lat = chem.data.sizes["latitude"]
        boxm_for.boxm.lon = chem.data.sizes["longitude"]
        boxm_for.boxm.alt = chem.data.sizes["level"]
        boxm_for.boxm.dts = 3600
        # Met variables
        boxm_for.boxm.temp = met["air_temperature"].data.values
        boxm_for.boxm.pressure = met["air_pressure"].data.values
        boxm_for.boxm.spec_hum = met["specific_humidity"].data.values
        boxm_for.boxm.m = M
        boxm_for.boxm.h2o = H2O
        boxm_for.boxm.o2 = O2
        print("met done")

        # Chem variables
        boxm_for.boxm.y = chem["Y"].data.values
        #boxm_for.boxm.yp = chem["YP"].data.values
        boxm_for.boxm.rc = chem["RC"].data.values
        boxm_for.boxm.dj = chem["DJ"].data.values
        boxm_for.boxm.em = chem["EM"].data.values
        boxm_for.boxm.fl = chem["FL"].data.values
        print("chem 5d done")
        boxm_for.boxm.j = chem["J"].data.values
        boxm_for.boxm.soa = chem["soa"].data.values
        boxm_for.boxm.mom = chem["mom"].data.values
        boxm_for.boxm.br01 = chem["BR01"].data.values
        boxm_for.boxm.ro2 = chem["RO2"].data.values

        print(boxm_for.boxm.__doc__)


    def _deriv(self, time_idx):
        boxm_for.boxm.deriv(int(time_idx+1))
   

    def _chemco(self, ts):
        """Calculate the thermal rate coefficients and reaction rates for each timestep."""

        N_A = 6.022e23 # Avogadro's number

        # Set met and chem according to self
        met = self.met
        chem = self.chem

        temp = met["air_temperature"].data.sel(time=ts)
        pressure = met["air_pressure"].data.sel(time=ts)
        spec_hum = met["specific_humidity"].data.sel(time=ts)

        # Get air density from pycontrails physics.thermo script
        rho_d = thermo.rho_d(temp, pressure)

        # Calculate number density of air (M) to feed into box model calcs
        M = (N_A / constants.M_d) * rho_d * 1e-6 # [molecules / cm^3]

        # Calculate H2O number concentration to feed into box model calcs
        H2O = (spec_hum / constants.M_v) * N_A * rho_d * 1e-6 # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        O2 = 2.079E-01*M
        N2 = 7.809E-01*M

        mom = chem["mom"].data.sel(time=ts)
        RO2 = chem["RO2"].data.sel(time=ts) 

        # SIMPLE RATE COEFFICIENTS         
        KRO2NO  = 2.54e-12*np.exp(360/temp) 
        KAPNO   = 8.10e-12*np.exp(270/temp) 
        KRO2NO3 = 2.50e-12 
        KRO2HO2 = 2.91e-13*np.exp(1300/temp) 
        KAPHO2  = 4.30e-13*np.exp(1040/temp) 
        KNO3AL  = 1.44e-12*np.exp(-1862/temp) 
        KDEC    = 1.0e+06
        KALKOXY = 3.70e-14*np.exp(-460/temp)*O2 
        KALKPXY = 1.80e-14*np.exp(-260/temp)*O2 
        BR01 = (0.156 + 9.77e+08*np.exp(-6415/temp)) 
        KIN = 6.2E-03 * mom
        KOUT2604 = 4.34*np.exp(-7776/(constants.R*temp))
        KOUT4608 = 4.34*np.exp(-9765/(constants.R*temp))
        KOUT2631 = 4.34*np.exp(-14500/(constants.R*temp))
        KOUT2635 = 4.34*np.exp(-12541/(constants.R*temp))
        KOUT4610 = 4.34*np.exp(-10513/(constants.R*temp))
        KOUT2605 = 4.34*np.exp(-8879/(constants.R*temp))
        KOUT2630 = 4.34*np.exp(-12639/(constants.R*temp))
        KOUT2629 = 4.34*np.exp(-4954/(constants.R*temp))
        KOUT2632 = 4.34*np.exp(-3801/(constants.R*temp))
        KOUT2637 = 4.34*np.exp(-16752/(constants.R*temp))
        KOUT3612 = 4.34*np.exp(-8362/(constants.R*temp))
        KOUT3613 = 4.34*np.exp(-11003/(constants.R*temp))
        KOUT3442 = 4.34*np.exp(-12763/(constants.R*temp))

        chem["BR01"].data.loc[:, :, :, ts] = BR01

        # COMPLEX RATE COEFFICIENTS                    
                                                                    
        # KFPAN
        KC0      = 2.70e-28*M*(temp/300)**-7.1 
        KCI      = 1.21e-11*(temp/300)**-0.9    
        KRC      = KC0/KCI    
        FCC      = 0.30       
        FC       = 10**(np.log10(FCC)/(1+(np.log10(KRC))**2)) 
        KFPAN    = (KC0*KCI)*FC/(KC0+KCI) 
                                                            
        # KBPAN
        KD0      = 4.90e-03*M*np.exp(-12100/temp) 
        KDI      = 3.70e+16*np.exp(-13600/temp)  
        KRD      = KD0/KDI    
        FCD      = 0.30       
        FD       = 10**(np.log10(FCD)/(1+(np.log10(KRD))**2)) 
        KBPAN    = (KD0*KDI)*FD/(KD0+KDI) 
                                                                
        # KMT01          
        K10      = 9.00e-32*M*(temp/300)**-1.5 
        K1I      = 3.00e-11*(temp/300)**0.3    
        KR1      = K10/K1I    
        FC1      = 0.6 
        F1       = 10**(np.log10(FC1)/(1+(np.log10(KR1))**2)) 
        KMT01    = (K10*K1I)*F1/(K10+K1I) 
                                                                            
        # KMT02                                                   
        K20      = 9.00e-32*((temp/300)**-2.0)*M 
        K2I      = 2.20e-11
        KR2      = K20/K2I    
        FC2      = 0.6 
        Fa2      = 10**(np.log10(FC2)/(1+(np.log10(KR2))**2)) 
        KMT02    = (K20*K2I)*Fa2/(K20+K2I) 
                                                                    
        # KMT03  : NO2 + NO3 = N2O5                               
        # IUPAC 2001                                                       
        K30      = 2.70e-30*M*(temp/300)**-3.4 
        K3I      = 2.00e-12*(temp/300)**0.2    
        KR3      = K30/K3I    
        FC3      = (np.exp(-temp/250) + np.exp(-1050/temp)) 
        F3       = 10**(np.log10(FC3)/(1+(np.log10(KR3))**2)) 
        KMT03    = (K30*K3I)*F3/(K30+K3I) 
                                                                    
        # KMT04  : N2O5 = NO2 + NO3                     
        # IUPAC 1997/2001                                                 
        K40      = (2.20e-03*M*(temp/300)**-4.34)*(np.exp(-11080/temp))
        K4I      = (9.70e+14*(temp/300)**0.1)*np.exp(-11080/temp)    
        KR4      = K40/K4I    
        FC4      = (np.exp(-temp/250) + np.exp(-1050/temp))
        Fa4      = 10**(np.log10(FC4)/(1+(np.log10(KR4))**2)) 
        KMT04    = (K40*K4I)*Fa4/(K40+K4I)       

        # KMT05                                                   
        KMT05    =  1 + ((0.6*M)/(2.687e+19*(273/temp))) 

        # KMT06                                                   
        KMT06    =  1 + (1.40e-21*np.exp(2200/temp)*H2O) 

        # KMT07 : OH + NO = HONO                              
        # IUPAC 2001                                                      
        K70      = 7.00e-31*M*(temp/300)**-2.6 
        K7I      = 3.60e-11*(temp/300)**0.1    
        KR7      = K70/K7I    
        FC7      = 0.6  
        F7       = 10**(np.log10(FC7)/(1+(np.log10(KR7))**2)) 
        KMT07    = (K70*K7I)*F7/(K70+K7I) 

        # NASA 2000                                                           

        # KMT08                                                    
        K80      = 2.50e-30*((temp/300)**-4.4)*M 
        K8I      = 1.60e-11 
        KR8      = K80/K8I 
        FC8      = 0.6 
        F8       = 10**(np.log10(FC8)/(1+(np.log10(KR8))**2)) 
        KMT08    = (K80*K8I)*F8/(K80+K8I) 

        # KMT09 : HO2 + NO2 = HO2NO2                            
        # IUPAC 1997/2001                                                 
        K90      = 1.80e-31*M*(temp/300)**-3.2 
        K9I      = 4.70e-12    
        KR9      = K90/K9I    
        FC9      = 0.6 
        F9       = 10**(np.log10(FC9)/(1+(np.log10(KR9))**2)) 
        KMT09    = (K90*K9I)*F9/(K90+K9I) 

        # KMT10 : HO2NO2 = HO2 + NO2                     
        # IUPAC 2001                                                      
        K100     = 4.10e-05*M*np.exp(-10650/temp) 
        K10I     = 5.70e+15*np.exp(-11170/temp)   
        KR10     = K100/K10I    
        FC10     = 0.5 
        F10      = 10**(np.log10(FC10)/(1+(np.log10(KR10))**2)) 
        KMT10    = (K100*K10I)*F10/(K100+K10I) 

        # KMT11 : OH + HNO3 = H2O + NO3                     
        # IUPAC 2001                                                      
        K1       = 7.20e-15*np.exp(785/temp) 
        K3       = 1.90e-33*np.exp(725/temp) 
        K4       = 4.10e-16*np.exp(1440/temp) 
        K2       = (K3*M)/(1+(K3*M/K4)) 
        KMT11    = K1 + K2 

        # KMT12 : OH + SO2 = HSO3                                  
        # IUPAC 2003                                                      
        K0       = 3.0e-31*((temp/300)**-3.3)*M 
        KI       = 1.5e-12 
        KR1      = K0/KI 
        FC       = 0.6 
        F        =10**(np.log10(FC)/(1+(np.log10(KR1))**2)) 
        KMT12    =(K0*KI*F)/(K0+KI) 

        # KMT13 : CH3O2 + NO2 = CH3O2NO2                           
        # IUPAC 2003                                                       
        K130     = 2.50e-30*((temp/300)**-5.5)*M 
        K13I     = 7.50e-12 
        KR13     = K130/K13I 
        FC13     = 0.36 
        F13      = 10**(np.log10(FC13)/(1+(np.log10(KR13))**2)) 
        KMT13    = (K130*K13I)*F13/(K130+K13I) 

        # KMT14 : CH3O2NO2 = CH3O2 + NO2                      
        # IUPAC 2001                                                       
        K140     = 9.00e-05*np.exp(-9690/temp)*M 
        K14I     = 1.10e+16*np.exp(-10560/temp) 
        KR14     = K140/K14I 
        FC14     = 0.36 
        F14      = 10**(np.log10(FC14)/(1+(np.log10(KR14))**2)) 
        KMT14    = (K140*K14I)*F14/(K140+K14I) 

        # KMT15 : OH + C2H4                                      
        # IUPAC 2001                                                      
        K150     = 6.00e-29*((temp/298)**-4.0)*M 
        K15I     = 9.00e-12*((temp/298)**-1.1) 
        KR15     = K150/K15I 
        FC15     = 0.7
        F15      = 10**(np.log10(FC15)/(1+(np.log10(KR15))**2)) 
        KMT15    = (K150*K15I)*F15/(K150+K15I) 

        # KMT16  :  OH  +  C3H6         
        # IUPAC 2003                                                     
        K160     = 3.00e-27*((temp/298)**-3.0)*M 
        K16I     = 2.80e-11*((temp/298)**-1.3) 
        KR16     = K160/K16I 
        FC16     = 0.5 
        F16      = 10**(np.log10(FC16)/(1+(np.log10(KR16))**2)) 
        KMT16    = (K160*K16I)*F16/(K160+K16I) 
                                                                    
        # KMT17                                                   
        K170     = 5.00e-30*((temp/298)**-1.5)*M 
        K17I     = 9.40e-12*np.exp(-700/temp) 
        KR17     = K170/K17I 
        FC17     = (np.exp(-temp/580) + np.exp(-2320/temp)) 
        F17      = 10**(np.log10(FC17)/(1+(np.log10(KR17))**2)) 
        KMT17    = (K170*K17I)*F17/(K170+K17I) 

        # LIST OF ALL REACTIONS
        # Reaction (1) O = O3 
        chem["RC"].data.loc[:, :, :, ts, 1] = 5.60e-34*O2*N2*((temp/300)**-2.6)
    
        # Reaction (2) O = O3                                                             
        chem["RC"].data.loc[:, :, :, ts, 2] = 6.00e-34*O2*O2*((temp/300)**-2.6)
        
        # Reaction (3) O + O3 =                                                           
        chem["RC"].data.loc[:, :, :, ts, 3] = 8.00e-12*np.exp(-2060/temp)         

        # Reaction (4) O + NO = NO2                                                       
        chem["RC"].data.loc[:, :, :, ts, 4] = KMT01                            

        # Reaction (5) O + NO2 = NO                                                       
        chem["RC"].data.loc[:, :, :, ts, 5] = 5.50e-12*np.exp(188/temp)           

        # Reaction (6) O + NO2 = NO3                                                      
        chem["RC"].data.loc[:, :, :, ts, 6] = KMT02                            

        # Reaction (7) O1D = O                                                            
        chem["RC"].data.loc[:, :, :, ts, 7] = 3.20e-11*O2*np.exp(67/temp)         

        # Reaction (8) O1D = O                                                            
        chem["RC"].data.loc[:, :, :, ts, 8] = 1.80e-11*N2*np.exp(107/temp)        

        # Reaction (9) NO + O3 = NO2                                                      
        chem["RC"].data.loc[:, :, :, ts, 9] = 1.40e-12*np.exp(-1310/temp)         

        # Reaction (10) NO2 + O3 = NO3                                                     
        chem["RC"].data.loc[:, :, :, ts, 10] = 1.40e-13*np.exp(-2470/temp)         

        # Reaction (11) NO + NO = NO2 + NO2                                                
        chem["RC"].data.loc[:, :, :, ts, 11] = 3.30e-39*np.exp(530/temp)*O2        

        # Reaction (12) NO + NO3 = NO2 + NO2                                               
        chem["RC"].data.loc[:, :, :, ts, 12] = 1.80e-11*np.exp(110/temp)           

        # Reaction (13) NO2 + NO3 = NO + NO2                                               
        chem["RC"].data.loc[:, :, :, ts, 13] = 4.50e-14*np.exp(-1260/temp)         

        # Reaction (14) NO2 + NO3 = N2O5                                                   
        chem["RC"].data.loc[:, :, :, ts, 14] = KMT03                            

        # Reaction (15) N2O5 = NO2 + NO3                                                   
        chem["RC"].data.loc[:, :, :, ts, 15] = KMT04                            

        # Reaction (16) O1D = OH + OH                                                      
        chem["RC"].data.loc[:, :, :, ts, 16] = 2.20e-10                     

        # Reaction (17) OH + O3 = HO2                                                      
        chem["RC"].data.loc[:, :, :, ts, 17] = 1.70e-12*np.exp(-940/temp)          

        # Reaction (18) OH + H2 = HO2                                                      
        chem["RC"].data.loc[:, :, :, ts, 18] = 7.70e-12*np.exp(-2100/temp)         

        # Reaction (19) OH + CO = HO2                                                      
        chem["RC"].data.loc[:, :, :, ts, 19] = 1.30e-13*KMT05                   

        # Reaction (20) OH + H2O2 = HO2                                                    
        chem["RC"].data.loc[:, :, :, ts, 20] = 2.90e-12*np.exp(-160/temp)          

        # Reaction (21) HO2 + O3 = OH                                                      
        chem["RC"].data.loc[:, :, :, ts, 21] = 2.03e-16*((temp/300)**4.57)*np.exp(693/temp)  

        # Reaction (22) OH + HO2 =                                                         
        chem["RC"].data.loc[:, :, :, ts, 22] = 4.80e-11*np.exp(250/temp)           

        # Reaction (23) HO2 + HO2 = H2O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 23] = 2.20e-13*KMT06*np.exp(600/temp)     

        # Reaction (24) HO2 + HO2 = H2O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 24] = 1.90e-33*M*KMT06*np.exp(980/temp)   

        # Reaction (25) OH + NO = HONO                                                     
        chem["RC"].data.loc[:, :, :, ts, 25] = KMT07                            

        # Reaction (26) NO2 = HONO                                                         
        chem["RC"].data.loc[:, :, :, ts, 26] = 0.0                          

        # Reaction (27) OH + NO2 = HNO3                                                    
        chem["RC"].data.loc[:, :, :, ts, 27] = KMT08                            

        # Reaction (28) OH + NO3 = HO2 + NO2                                               
        chem["RC"].data.loc[:, :, :, ts, 28] = 2.00e-11                         

        # Reaction (29) HO2 + NO = OH + NO2                                                
        chem["RC"].data.loc[:, :, :, ts, 29] = 3.60e-12*np.exp(270/temp)           

        # Reaction (30) HO2 + NO2 = HO2NO2                                                 
        chem["RC"].data.loc[:, :, :, ts, 30] = KMT09                            

        # Reaction (31) HO2NO2 = HO2 + NO2                                                 
        chem["RC"].data.loc[:, :, :, ts, 31] = KMT10                            

        # Reaction (32) OH + HO2NO2 = NO2                                                  
        chem["RC"].data.loc[:, :, :, ts, 32] = 1.90e-12*np.exp(270/temp)           

        # Reaction (33) HO2 + NO3 = OH + NO2                                               
        chem["RC"].data.loc[:, :, :, ts, 33] = 4.00e-12                         

        # Reaction (34) OH + HONO = NO2                                                    
        chem["RC"].data.loc[:, :, :, ts, 34] = 2.50e-12*np.exp(-260/temp)          

        # Reaction (35) OH + HNO3 = NO3                                                    
        chem["RC"].data.loc[:, :, :, ts, 35] = KMT11                            

        # Reaction (36) O + SO2 = SO3                                                      
        chem["RC"].data.loc[:, :, :, ts, 36] = 4.00e-32*np.exp(-1000/temp)*M       

        # Reaction (37) OH + SO2 = HSO3                                                    
        chem["RC"].data.loc[:, :, :, ts, 37] = KMT12                            

        # Reaction (38) HSO3 = HO2 + SO3                                                   
        chem["RC"].data.loc[:, :, :, ts, 38] = 1.30e-12*np.exp(-330/temp)*O2       

        # Reaction (39) HNO3 = NA                                                          
        chem["RC"].data.loc[:, :, :, ts, 39] = 0.0                         

        # Reaction (40) N2O5 = NA + NA                                                     
        chem["RC"].data.loc[:, :, :, ts, 40] = 0.0                         

        # Reaction (41) SO3 = SA                                                           
        chem["RC"].data.loc[:, :, :, ts, 41] = 1.20e-15*H2O                     

        # Reaction (42) OH + CH4 = CH3O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 42] = 9.65e-20*temp**2.58*np.exp(-1082/temp) 

        # Reaction (43) OH + C2H6 = C2H5O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 43] = 1.52e-17*temp**2*np.exp(-498/temp) 

        # Reaction (44) OH + C3H8 = IC3H7O2                                                
        chem["RC"].data.loc[:, :, :, ts, 44] = 1.55e-17*temp**2*np.exp(-61/temp)*0.736  

        # Reaction (45) OH + C3H8 = RN10O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 45] = 1.55e-17*temp**2*np.exp(-61/temp)*0.264  

        # Reaction (46) OH + NC4H10 = RN13O2                                               
        chem["RC"].data.loc[:, :, :, ts, 46] = 1.69e-17*temp**2*np.exp(145/temp)  

        # Reaction (47) OH + C2H4 = HOCH2CH2O2                                             
        chem["RC"].data.loc[:, :, :, ts, 47] = KMT15                        

        # Reaction (48) OH + C3H6 = RN9O2                                                  
        chem["RC"].data.loc[:, :, :, ts, 48] = KMT16                        

        # Reaction (49) OH + TBUT2ENE = RN12O2                                             
        chem["RC"].data.loc[:, :, :, ts, 49] = 1.01e-11*np.exp(550/temp)       

        # Reaction (50) NO3 + C2H4 = NRN6O2                                                
        chem["RC"].data.loc[:, :, :, ts, 50] = 2.10e-16                     

        # Reaction (51) NO3 + C3H6 = NRN9O2                                                
        chem["RC"].data.loc[:, :, :, ts, 51] = 9.40e-15                     

        # Reaction (52) NO3 + TBUT2ENE = NRN12O2                                           
        chem["RC"].data.loc[:, :, :, ts, 52] = 3.90e-13                     

        # Reaction (53) O3 + C2H4 = HCHO + CO + HO2 + OH                                   
        chem["RC"].data.loc[:, :, :, ts, 53] = 9.14e-15*np.exp(-2580/temp)*0.13  

        # Reaction (54) O3 + C2H4 = HCHO + HCOOH                                           
        chem["RC"].data.loc[:, :, :, ts, 54] = 9.14e-15*np.exp(-2580/temp)*0.87  

        # Reaction (55) O3 + C3H6 = HCHO + CO + CH3O2 + OH                                 
        chem["RC"].data.loc[:, :, :, ts, 55] = 5.51e-15*np.exp(-1878/temp)*0.36  

        # Reaction (56) O3 + C3H6 = HCHO + CH3CO2H                                         
        chem["RC"].data.loc[:, :, :, ts, 56] = 5.51e-15*np.exp(-1878/temp)*0.64  

        # Reaction (57) O3 + TBUT2ENE = CH3CHO + CO + CH3O2 + OH                           
        chem["RC"].data.loc[:, :, :, ts, 57] = 6.64e-15*np.exp(-1059/temp)*0.69 

        # Reaction (58) O3 + TBUT2ENE = CH3CHO + CH3CO2H                                   
        chem["RC"].data.loc[:, :, :, ts, 58] = 6.64e-15*np.exp(-1059/temp)*0.31 

        # Reaction (59) OH + C5H8 = RU14O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 59] = 2.54e-11*np.exp(410/temp)       

        # Reaction (60) NO3 + C5H8 = NRU14O2                                               
        chem["RC"].data.loc[:, :, :, ts, 60] = 3.03e-12*np.exp(-446/temp)      

        # Reaction (61) O3 + C5H8 = UCARB10 + CO + HO2 + OH                                
        chem["RC"].data.loc[:, :, :, ts, 61] = 7.86e-15*np.exp(-1913/temp)*0.27 

        # Reaction (62) O3 + C5H8 = UCARB10 + HCOOH                                        
        chem["RC"].data.loc[:, :, :, ts, 62] = 7.86e-15*np.exp(-1913/temp)*0.73 

        # Reaction (63) APINENE + OH = RTN28O2                                             
        chem["RC"].data.loc[:, :, :, ts, 63] = 1.20e-11*np.exp(444/temp)           

        # Reaction (64) APINENE + NO3 = NRTN28O2                                           
        chem["RC"].data.loc[:, :, :, ts, 64] = 1.19e-12*np.exp(490/temp)           

        # Reaction (65) APINENE + O3 = OH + RTN26O2                                        
        chem["RC"].data.loc[:, :, :, ts, 65] = 1.01e-15*np.exp(-732/temp)*0.80  

        # Reaction (66) APINENE + O3 = TNCARB26 + H2O2                                     
        chem["RC"].data.loc[:, :, :, ts, 66] = 1.01e-15*np.exp(-732/temp)*0.075  

        # Reaction (67) APINENE + O3 = RCOOH25                                             
        chem["RC"].data.loc[:, :, :, ts, 67] = 1.01e-15*np.exp(-732/temp)*0.125  

        # Reaction (68) BPINENE + OH = RTX28O2                                             
        chem["RC"].data.loc[:, :, :, ts, 68] = 2.38e-11*np.exp(357/temp) 

        # Reaction (69) BPINENE + NO3 = NRTX28O2                                           
        chem["RC"].data.loc[:, :, :, ts, 69] = 2.51e-12 

        # Reaction (70) BPINENE + O3 =  RTX24O2 + OH                                       
        chem["RC"].data.loc[:, :, :, ts, 70] = 1.50e-17*0.35 

        # Reaction (71) BPINENE + O3 =  HCHO + TXCARB24 + H2O2                             
        chem["RC"].data.loc[:, :, :, ts, 71] = 1.50e-17*0.20 

        # Reaction (72) BPINENE + O3 =  HCHO + TXCARB22                                    
        chem["RC"].data.loc[:, :, :, ts, 72] = 1.50e-17*0.25 

        # Reaction (73) BPINENE + O3 =  TXCARB24 + CO                                      
        chem["RC"].data.loc[:, :, :, ts, 73] = 1.50e-17*0.20 

        # Reaction (74) C2H2 + OH = HCOOH + CO + HO2                                       
        chem["RC"].data.loc[:, :, :, ts, 74] = KMT17*0.364 

        # Reaction (75) C2H2 + OH = CARB3 + OH                                             
        chem["RC"].data.loc[:, :, :, ts, 75] = KMT17*0.636 

        # Reaction (76) BENZENE + OH = RA13O2                                              
        chem["RC"].data.loc[:, :, :, ts, 76] = 2.33e-12*np.exp(-193/temp)*0.47 

        # Reaction (77) BENZENE + OH = AROH14 + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 77] = 2.33e-12*np.exp(-193/temp)*0.53 

        # Reaction (78) TOLUENE + OH = RA16O2                                              
        chem["RC"].data.loc[:, :, :, ts, 78] = 1.81e-12*np.exp(338/temp)*0.82 

        # Reaction (79) TOLUENE + OH = AROH17 + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 79] = 1.81e-12*np.exp(338/temp)*0.18 

        # Reaction (80) OXYL + OH = RA19AO2                                                
        chem["RC"].data.loc[:, :, :, ts, 80] = 1.36e-11*0.70 

        # Reaction (81) OXYL + OH = RA19CO2                                                
        chem["RC"].data.loc[:, :, :, ts, 81] = 1.36e-11*0.30 

        # Reaction (82) OH + HCHO = HO2 + CO                                               
        chem["RC"].data.loc[:, :, :, ts, 82] = 1.20e-14*temp*np.exp(287/temp)  

        # Reaction (83) OH + CH3CHO = CH3CO3                                               
        chem["RC"].data.loc[:, :, :, ts, 83] = 5.55e-12*np.exp(311/temp)             

        # Reaction (84) OH + C2H5CHO = C2H5CO3                                             
        chem["RC"].data.loc[:, :, :, ts, 84] = 1.96e-11                                

        # Reaction (85) NO3 + HCHO = HO2 + CO + HNO3                                       
        chem["RC"].data.loc[:, :, :, ts, 85] = 5.80e-16                  

        # Reaction (86) NO3 + CH3CHO = CH3CO3 + HNO3                                       
        chem["RC"].data.loc[:, :, :, ts, 86] = KNO3AL                   

        # Reaction (87) NO3 + C2H5CHO = C2H5CO3 + HNO3                                     
        chem["RC"].data.loc[:, :, :, ts, 87] = KNO3AL*2.4             

        # Reaction (88) OH + CH3COCH3 = RN8O2                                              
        chem["RC"].data.loc[:, :, :, ts, 88] = 5.34e-18*temp**2*np.exp(-230/temp) 

        # Reaction (89) MEK + OH = RN11O2                                                  
        chem["RC"].data.loc[:, :, :, ts, 89] = 3.24e-18*temp**2*np.exp(414/temp)

        # Reaction (90) OH + CH3OH = HO2 + HCHO                                            
        chem["RC"].data.loc[:, :, :, ts, 90] = 6.01e-18*temp**2*np.exp(170/temp)  

        # Reaction (91) OH + C2H5OH = CH3CHO + HO2                                         
        chem["RC"].data.loc[:, :, :, ts, 91] = 6.18e-18*temp**2*np.exp(532/temp)*0.887 

        # Reaction (92) OH + C2H5OH = HOCH2CH2O2                                           
        chem["RC"].data.loc[:, :, :, ts, 92] = 6.18e-18*temp**2*np.exp(532/temp)*0.113 

        # Reaction (93) NPROPOL + OH = C2H5CHO + HO2                                       
        chem["RC"].data.loc[:, :, :, ts, 93] = 5.53e-12*0.49 

        # Reaction (94) NPROPOL + OH = RN9O2                                               
        chem["RC"].data.loc[:, :, :, ts, 94] = 5.53e-12*0.51 

        # Reaction (95) OH + IPROPOL = CH3COCH3 + HO2                                      
        chem["RC"].data.loc[:, :, :, ts, 95] = 4.06e-18*temp**2*np.exp(788/temp)*0.86 

        # Reaction (96) OH + IPROPOL = RN9O2                                               
        chem["RC"].data.loc[:, :, :, ts, 96] = 4.06e-18*temp**2*np.exp(788/temp)*0.14 

        # Reaction (97) HCOOH + OH = HO2                                                   
        chem["RC"].data.loc[:, :, :, ts, 97] = 4.50e-13 

        # Reaction (98) CH3CO2H + OH = CH3O2                                               
        chem["RC"].data.loc[:, :, :, ts, 98] = 8.00e-13 

        # Reaction (99) OH + CH3CL = CH3O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 99] = 7.33e-18*temp**2*np.exp(-809/temp)   

        # Reaction (100) OH + CH2CL2 = CH3O2                                                
        chem["RC"].data.loc[:, :, :, ts, 100] = 6.14e-18*temp**2*np.exp(-389/temp)   

        # Reaction (101) OH + CHCL3 = CH3O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 101] = 1.80e-18*temp**2*np.exp(-129/temp)   

        # Reaction (102) OH + CH3CCL3 = C2H5O2                                              
        chem["RC"].data.loc[:, :, :, ts, 102] = 2.25e-18*temp**2*np.exp(-910/temp)   

        # Reaction (103) OH + TCE = HOCH2CH2O2                                              
        chem["RC"].data.loc[:, :, :, ts, 103] = 9.64e-12*np.exp(-1209/temp)         

        # Reaction (104) OH + TRICLETH = HOCH2CH2O2                                         
        chem["RC"].data.loc[:, :, :, ts, 104] = 5.63e-13*np.exp(427/temp)            

        # Reaction (105) OH + CDICLETH = HOCH2CH2O2                                         
        chem["RC"].data.loc[:, :, :, ts, 105] = 1.94e-12*np.exp(90/temp)            

        # Reaction (106) OH + TDICLETH = HOCH2CH2O2                                         
        chem["RC"].data.loc[:, :, :, ts, 106] = 1.01e-12*np.exp(250/temp)           

        # Reaction (107) CH3O2 + NO = HCHO + HO2 + NO2                                      
        chem["RC"].data.loc[:, :, :, ts, 107] = 3.00e-12*np.exp(280/temp)*0.999 

        # Reaction (108) C2H5O2 + NO = CH3CHO + HO2 + NO2                                   
        chem["RC"].data.loc[:, :, :, ts, 108] = 2.60e-12*np.exp(365/temp)*0.991 

        # Reaction (109) RN10O2 + NO = C2H5CHO + HO2 + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 109] = 2.80e-12*np.exp(360/temp)*0.980 

        # Reaction (110) IC3H7O2 + NO = CH3COCH3 + HO2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 110] = 2.70e-12*np.exp(360/temp)*0.958 

        # Reaction (111) RN13O2 + NO = CH3CHO + C2H5O2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 111] = KRO2NO*0.917*BR01       

        # Reaction (112) RN13O2 + NO = CARB11A + HO2 + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 112] = KRO2NO*0.917*(1-BR01)   

        # Reaction (113) RN16O2 + NO = RN15AO2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 113] = KRO2NO*0.877                 

        # Reaction (114) RN19O2 + NO = RN18AO2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 114] = KRO2NO*0.788                 

        # Reaction (115) RN13AO2 + NO = RN12O2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 115] = KRO2NO                       

        # Reaction (116) RN16AO2 + NO = RN15O2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 116] = KRO2NO                       

        # Reaction (117) RA13O2 + NO = CARB3 + UDCARB8 + HO2 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 117] = KRO2NO*0.918       

        # Reaction (118) RA16O2 + NO = CARB3 + UDCARB11 + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 118] = KRO2NO*0.889*0.7 

        # Reaction (119) RA16O2 + NO = CARB6 + UDCARB8 + HO2 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 119] = KRO2NO*0.889*0.3 

        # Reaction (120) RA19AO2 + NO = CARB3 + UDCARB14 + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 120] = KRO2NO*0.862       

        # Reaction (121) RA19CO2 + NO = CARB9 + UDCARB8 + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 121] = KRO2NO*0.862       

        # Reaction (122) HOCH2CH2O2 + NO = HCHO + HCHO + HO2 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 122] = KRO2NO*0.995*0.776  

        # Reaction (123) HOCH2CH2O2 + NO = HOCH2CHO + HO2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 123] = KRO2NO*0.995*0.224  

        # Reaction (124) RN9O2 + NO = CH3CHO + HCHO + HO2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 124] = KRO2NO*0.979     

        # Reaction (125) RN12O2 + NO = CH3CHO + CH3CHO + HO2 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 125] = KRO2NO*0.959     

        # Reaction (126) RN15O2 + NO = C2H5CHO + CH3CHO + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 126] = KRO2NO*0.936     

        # Reaction (127) RN18O2 + NO = C2H5CHO + C2H5CHO + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 127] = KRO2NO*0.903     

        # Reaction (128) RN15AO2 + NO = CARB13 + HO2 + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 128] = KRO2NO*0.975     

        # Reaction (129) RN18AO2 + NO = CARB16 + HO2 + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 129] = KRO2NO*0.946     

        # Reaction (130) CH3CO3 + NO = CH3O2 + NO2                                          
        chem["RC"].data.loc[:, :, :, ts, 130] = KAPNO                      

        # Reaction (131) C2H5CO3 + NO = C2H5O2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 131] = KAPNO                      

        # Reaction (132) HOCH2CO3 + NO = HO2 + HCHO + NO2                                   
        chem["RC"].data.loc[:, :, :, ts, 132] = KAPNO                      

        # Reaction (133) RN8O2 + NO = CH3CO3 + HCHO + NO2                                   
        chem["RC"].data.loc[:, :, :, ts, 133] = KRO2NO                     

        # Reaction (134) RN11O2 + NO = CH3CO3 + CH3CHO + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 134] = KRO2NO                     

        # Reaction (135) RN14O2 + NO = C2H5CO3 + CH3CHO + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 135] = KRO2NO                     

        # Reaction (136) RN17O2 + NO = RN16AO2 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 136] = KRO2NO                     

        # Reaction (137) RU14O2 + NO = UCARB12 + HO2 +  NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 137] = KRO2NO*0.900*0.252  

        # Reaction (138) RU14O2 + NO = UCARB10 + HCHO + HO2 + NO2                           
        chem["RC"].data.loc[:, :, :, ts, 138] = KRO2NO*0.900*0.748 

        # Reaction (139) RU12O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 139] = KRO2NO*0.7         

        # Reaction (140) RU12O2 + NO = CARB7 + CO + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 140] = KRO2NO*0.3         

        # Reaction (141) RU10O2 + NO = CH3CO3 + HOCH2CHO + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 141] = KRO2NO*0.5         

        # Reaction (142) RU10O2 + NO = CARB6 + HCHO + HO2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 142] = KRO2NO*0.3         

        # Reaction (143) RU10O2 + NO = CARB7 + HCHO + HO2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 143] = KRO2NO*0.2          

        # Reaction (144) NRN6O2 + NO = HCHO + HCHO + NO2 + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 144] = KRO2NO                 

        # Reaction (145) NRN9O2 + NO = CH3CHO + HCHO + NO2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 145] = KRO2NO                 

        # Reaction (146) NRN12O2 + NO = CH3CHO + CH3CHO + NO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 146] = KRO2NO                 

        # Reaction (147) NRU14O2 + NO = NUCARB12 + HO2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 147] = KRO2NO                 

        # Reaction (148) NRU12O2 + NO = NOA + CO + HO2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 148] = KRO2NO                 

        # Reaction (149) RTN28O2 + NO = TNCARB26 + HO2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 149] = KRO2NO*0.767*0.915  

        # Reaction (150) RTN28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 150] = KRO2NO*0.767*0.085  

        # Reaction (151) NRTN28O2 + NO = TNCARB26 + NO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 151] = KRO2NO                  

        # Reaction (152) RTN26O2 + NO = RTN25O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 152] = KAPNO                   

        # Reaction (153) RTN25O2 + NO = RTN24O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 153] = KRO2NO*0.840        

        # Reaction (154) RTN24O2 + NO = RTN23O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 154] = KRO2NO                   

        # Reaction (155) RTN23O2 + NO = CH3COCH3 + RTN14O2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 155] = KRO2NO                  

        # Reaction (156) RTN14O2 + NO = HCHO + TNCARB10 + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 156] = KRO2NO               

        # Reaction (157) RTN10O2 + NO = RN8O2 + CO + NO2                                    
        chem["RC"].data.loc[:, :, :, ts, 157] = KRO2NO               

        # Reaction (158) RTX28O2 + NO = TXCARB24 + HCHO + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 158] = KRO2NO*0.767*0.915  

        # Reaction (159) RTX28O2 + NO = CH3COCH3 + RN19O2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 159] = KRO2NO*0.767*0.085  

        # Reaction (160) NRTX28O2 + NO = TXCARB24 + HCHO + NO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 160] = KRO2NO            

        # Reaction (161) RTX24O2 + NO = TXCARB22 + HO2 + NO2                                
        chem["RC"].data.loc[:, :, :, ts, 161] = KRO2NO*0.843*0.6  

        # Reaction (162) RTX24O2 + NO = CH3COCH3 + RN13AO2 + HCHO + NO2                     
        chem["RC"].data.loc[:, :, :, ts, 162] = KRO2NO*0.843*0.4  

        # Reaction (163) RTX22O2 + NO = CH3COCH3 + RN13O2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 163] = KRO2NO*0.700         

        # Reaction (164) CH3O2 + NO2 = CH3O2NO2                                      
        chem["RC"].data.loc[:, :, :, ts, 164] = KMT13         

        # Reaction (165) CH3O2NO2 = CH3O2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 165] = KMT14         

        # Reaction (166) CH3O2 + NO = CH3NO3                                                
        chem["RC"].data.loc[:, :, :, ts, 166] = 3.00e-12*np.exp(280/temp)*0.001 

        # Reaction (167) C2H5O2 + NO = C2H5NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 167] = 2.60e-12*np.exp(365/temp)*0.009 

        # Reaction (168) RN10O2 + NO = RN10NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 168] = 2.80e-12*np.exp(360/temp)*0.020 

        # Reaction (169) IC3H7O2 + NO = IC3H7NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 169] = 2.70e-12*np.exp(360/temp)*0.042 

        # Reaction (170) RN13O2 + NO = RN13NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 170] = KRO2NO*0.083                 

        # Reaction (171) RN16O2 + NO = RN16NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 171] = KRO2NO*0.123                 

        # Reaction (172) RN19O2 + NO = RN19NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 172] = KRO2NO*0.212                 

        # Reaction (173) HOCH2CH2O2 + NO = HOC2H4NO3                                        
        chem["RC"].data.loc[:, :, :, ts, 173] = KRO2NO*0.005                 

        # Reaction (174) RN9O2 + NO = RN9NO3                                                
        chem["RC"].data.loc[:, :, :, ts, 174] = KRO2NO*0.021                 

        # Reaction (175) RN12O2 + NO = RN12NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 175] = KRO2NO*0.041                 

        # Reaction (176) RN15O2 + NO = RN15NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 176] = KRO2NO*0.064                 

        # Reaction (177) RN18O2 + NO = RN18NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 177] = KRO2NO*0.097                 

        # Reaction (178) RN15AO2 + NO = RN15NO3                                             
        chem["RC"].data.loc[:, :, :, ts, 178] = KRO2NO*0.025                 

        # Reaction (179) RN18AO2 + NO = RN18NO3                                             
        chem["RC"].data.loc[:, :, :, ts, 179] = KRO2NO*0.054                 

        # Reaction (180) RU14O2 + NO = RU14NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 180] = KRO2NO*0.100                 

        # Reaction (181) RA13O2 + NO = RA13NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 181] = KRO2NO*0.082                 

        # Reaction (182) RA16O2 + NO = RA16NO3                                              
        chem["RC"].data.loc[:, :, :, ts, 182] = KRO2NO*0.111                 

        # Reaction (183) RA19AO2 + NO = RA19NO3                                             
        chem["RC"].data.loc[:, :, :, ts, 183] = KRO2NO*0.138                 

        # Reaction (184) RA19CO2 + NO = RA19NO3                                             
        chem["RC"].data.loc[:, :, :, ts, 184] = KRO2NO*0.138                 

        # Reaction (185) RTN28O2 + NO = RTN28NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 185] = KRO2NO*0.233        

        # Reaction (186) RTN25O2 + NO = RTN25NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 186] = KRO2NO*0.160        

        # Reaction (187) RTX28O2 + NO = RTX28NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 187] = KRO2NO*0.233        

        # Reaction (188) RTX24O2 + NO = RTX24NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 188] = KRO2NO*0.157        

        # Reaction (189) RTX22O2 + NO = RTX22NO3                                            
        chem["RC"].data.loc[:, :, :, ts, 189] = KRO2NO*0.300        

        # Reaction (190) CH3O2 + NO3 = HCHO + HO2 + NO2                                     
        chem["RC"].data.loc[:, :, :, ts, 190] = KRO2NO3*0.40          

        # Reaction (191) C2H5O2 + NO3 = CH3CHO + HO2 + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 191] = KRO2NO3               

        # Reaction (192) RN10O2 + NO3 = C2H5CHO + HO2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 192] = KRO2NO3               

        # Reaction (193) IC3H7O2 + NO3 = CH3COCH3 + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 193] = KRO2NO3               

        # Reaction (194) RN13O2 + NO3 = CH3CHO + C2H5O2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 194] = KRO2NO3*BR01     

        # Reaction (195) RN13O2 + NO3 = CARB11A + HO2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 195] = KRO2NO3*(1-BR01) 

        # Reaction (196) RN16O2 + NO3 = RN15AO2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 196] = KRO2NO3               

        # Reaction (197) RN19O2 + NO3 = RN18AO2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 197] = KRO2NO3               

        # Reaction (198) RN13AO2 + NO3 = RN12O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 198] = KRO2NO3                      

        # Reaction (199) RN16AO2 + NO3 = RN15O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 199] = KRO2NO3                      

        # Reaction (200) RA13O2 + NO3 = CARB3 + UDCARB8 + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 200] = KRO2NO3            

        # Reaction (201) RA16O2 + NO3 = CARB3 + UDCARB11 + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 201] = KRO2NO3*0.7     

        # Reaction (202) RA16O2 + NO3 = CARB6 + UDCARB8 + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 202] = KRO2NO3*0.3     

        # Reaction (203) RA19AO2 + NO3 = CARB3 + UDCARB14 + HO2 + NO2                       
        chem["RC"].data.loc[:, :, :, ts, 203] = KRO2NO3           

        # Reaction (204) RA19CO2 + NO3 = CARB9 + UDCARB8 + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 204] = KRO2NO3           

        # Reaction (205) HOCH2CH2O2 + NO3 = HCHO + HCHO + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 205] = KRO2NO3*0.776  

        # Reaction (206) HOCH2CH2O2 + NO3 = HOCH2CHO + HO2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 206] = KRO2NO3*0.224  

        # Reaction (207) RN9O2 + NO3 = CH3CHO + HCHO + HO2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 207] = KRO2NO3         

        # Reaction (208) RN12O2 + NO3 = CH3CHO + CH3CHO + HO2 + NO2                         
        chem["RC"].data.loc[:, :, :, ts, 208] = KRO2NO3         

        # Reaction (209) RN15O2 + NO3 = C2H5CHO + CH3CHO + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 209] = KRO2NO3         

        # Reaction (210) RN18O2 + NO3 = C2H5CHO + C2H5CHO + HO2 + NO2                       
        chem["RC"].data.loc[:, :, :, ts, 210] = KRO2NO3         

        # Reaction (211) RN15AO2 + NO3 = CARB13 + HO2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 211] = KRO2NO3         

        # Reaction (212) RN18AO2 + NO3 = CARB16 + HO2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 212] = KRO2NO3         

        # Reaction (213) CH3CO3 + NO3 = CH3O2 + NO2                                         
        chem["RC"].data.loc[:, :, :, ts, 213] = KRO2NO3*1.60          

        # Reaction (214) C2H5CO3 + NO3 = C2H5O2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 214] = KRO2NO3*1.60          

        # Reaction (215) HOCH2CO3 + NO3 = HO2 + HCHO + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 215] = KRO2NO3*1.60         

        # Reaction (216) RN8O2 + NO3 = CH3CO3 + HCHO + NO2                                  
        chem["RC"].data.loc[:, :, :, ts, 216] = KRO2NO3               

        # Reaction (217) RN11O2 + NO3 = CH3CO3 + CH3CHO + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 217] = KRO2NO3               

        # Reaction (218) RN14O2 + NO3 = C2H5CO3 + CH3CHO + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 218] = KRO2NO3               

        # Reaction (219) RN17O2 + NO3 = RN16AO2 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 219] = KRO2NO3               

        # Reaction (220) RU14O2 + NO3 = UCARB12 + HO2 + NO2                                 
        chem["RC"].data.loc[:, :, :, ts, 220] = KRO2NO3*0.252     

        # Reaction (221) RU14O2 + NO3 = UCARB10 + HCHO + HO2 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 221] = KRO2NO3*0.748     

        # Reaction (222) RU12O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 222] = KRO2NO3*0.7         

        # Reaction (223) RU12O2 + NO3 = CARB7 + CO + HO2 + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 223] = KRO2NO3*0.3         

        # Reaction (224) RU10O2 + NO3 = CH3CO3 + HOCH2CHO + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 224] = KRO2NO3*0.5         

        # Reaction (225) RU10O2 + NO3 = CARB6 + HCHO + HO2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 225] = KRO2NO3*0.3         

        # Reaction (226) RU10O2 + NO3 = CARB7 + HCHO + HO2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 226] = KRO2NO3*0.2         

        # Reaction (227) NRN6O2 + NO3 = HCHO + HCHO + NO2 + NO2                             
        chem["RC"].data.loc[:, :, :, ts, 227] = KRO2NO3               

        # Reaction (228) NRN9O2 + NO3 = CH3CHO + HCHO + NO2 + NO2                           
        chem["RC"].data.loc[:, :, :, ts, 228] = KRO2NO3               

        # Reaction (229) NRN12O2 + NO3 = CH3CHO + CH3CHO + NO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 229] = KRO2NO3               

        # Reaction (230) NRU14O2 + NO3 = NUCARB12 + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 230] = KRO2NO3               

        # Reaction (231) NRU12O2 + NO3 = NOA + CO + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 231] = KRO2NO3               

        # Reaction (232) RTN28O2 + NO3 = TNCARB26 + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 232] = KRO2NO3                

        # Reaction (233) NRTN28O2 + NO3 = TNCARB26 + NO2 + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 233] = KRO2NO3                

        # Reaction (234) RTN26O2 + NO3 = RTN25O2 + NO2                                      
        chem["RC"].data.loc[:, :, :, ts, 234] = KRO2NO3*1.60                   

        # Reaction (235) RTN25O2 + NO3 = RTN24O2 + NO2                                      
        chem["RC"].data.loc[:, :, :, ts, 235] = KRO2NO3                 

        # Reaction (236) RTN24O2 + NO3 = RTN23O2 + NO2                                      
        chem["RC"].data.loc[:, :, :, ts, 236] = KRO2NO3                   

        # Reaction (237) RTN23O2 + NO3 = CH3COCH3 + RTN14O2 + NO2                           
        chem["RC"].data.loc[:, :, :, ts, 237] = KRO2NO3                 

        # Reaction (238) RTN14O2 + NO3 = HCHO + TNCARB10 + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 238] = KRO2NO3             

        # Reaction (239) RTN10O2 + NO3 = RN8O2 + CO + NO2                                   
        chem["RC"].data.loc[:, :, :, ts, 239] = KRO2NO3               

        # Reaction (240) RTX28O2 + NO3 = TXCARB24 + HCHO + HO2 + NO2                        
        chem["RC"].data.loc[:, :, :, ts, 240] = KRO2NO3             

        # Reaction (241) RTX24O2 + NO3 = TXCARB22 + HO2 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 241] = KRO2NO3             

        # Reaction (242) RTX22O2 + NO3 = CH3COCH3 + RN13O2 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 242] = KRO2NO3             

        # Reaction (243) NRTX28O2 + NO3 = TXCARB24 + HCHO + NO2 + NO2                       
        chem["RC"].data.loc[:, :, :, ts, 243] = KRO2NO3            

        # Reaction (244) CH3O2 + HO2 = CH3OOH                                               
        chem["RC"].data.loc[:, :, :, ts, 244] = 4.10e-13*np.exp(790/temp)  

        # Reaction (245) C2H5O2 + HO2 = C2H5OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 245] = 7.50e-13*np.exp(700/temp)  

        # Reaction (246) RN10O2 + HO2 = RN10OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 246] = KRO2HO2*0.520           

        # Reaction (247) IC3H7O2 + HO2 = IC3H7OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 247] = KRO2HO2*0.520           

        # Reaction (248) RN13O2 + HO2 = RN13OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 248] = KRO2HO2*0.625           

        # Reaction (249) RN16O2 + HO2 = RN16OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 249] = KRO2HO2*0.706           

        # Reaction (250) RN19O2 + HO2 = RN19OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 250] = KRO2HO2*0.770           

        # Reaction (251) RN13AO2 + HO2 = RN13OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 251] = KRO2HO2*0.625           

        # Reaction (252) RN16AO2 + HO2 = RN16OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 252] = KRO2HO2*0.706           

        # Reaction (253) RA13O2 + HO2 = RA13OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 253] = KRO2HO2*0.770           

        # Reaction (254) RA16O2 + HO2 = RA16OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 254] = KRO2HO2*0.820           

        # Reaction (255) RA19AO2 + HO2 = RA19OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 255] = KRO2HO2*0.859           

        # Reaction (256) RA19CO2 + HO2 = RA19OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 256] = KRO2HO2*0.859           

        # Reaction (257) HOCH2CH2O2 + HO2 = HOC2H4OOH                                       
        chem["RC"].data.loc[:, :, :, ts, 257] = 2.03e-13*np.exp(1250/temp) 

        # Reaction (258) RN9O2 + HO2 = RN9OOH                                               
        chem["RC"].data.loc[:, :, :, ts, 258] = KRO2HO2*0.520           

        # Reaction (259) RN12O2 + HO2 = RN12OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 259] = KRO2HO2*0.625           

        # Reaction (260) RN15O2 + HO2 = RN15OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 260] = KRO2HO2*0.706           

        # Reaction (261) RN18O2 + HO2 = RN18OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 261] = KRO2HO2*0.770           

        # Reaction (262) RN15AO2 + HO2 = RN15OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 262] = KRO2HO2*0.706           

        # Reaction (263) RN18AO2 + HO2 = RN18OOH                                            
        chem["RC"].data.loc[:, :, :, ts, 263] = KRO2HO2*0.770           

        # Reaction (264) CH3CO3 + HO2 = CH3CO3H                                             
        chem["RC"].data.loc[:, :, :, ts, 264] = KAPHO2                  

        # Reaction (265) C2H5CO3 + HO2 = C2H5CO3H                                           
        chem["RC"].data.loc[:, :, :, ts, 265] = KAPHO2                  

        # Reaction (266) HOCH2CO3 + HO2 = HOCH2CO3H                                         
        chem["RC"].data.loc[:, :, :, ts, 266] = KAPHO2                  

        # Reaction (267) RN8O2 + HO2 = RN8OOH                                               
        chem["RC"].data.loc[:, :, :, ts, 267] = KRO2HO2*0.520           

        # Reaction (268) RN11O2 + HO2 = RN11OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 268] = KRO2HO2*0.625           

        # Reaction (269) RN14O2 + HO2 = RN14OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 269] = KRO2HO2*0.706           

        # Reaction (270) RN17O2 + HO2 = RN17OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 270] = KRO2HO2*0.770           

        # Reaction (271) RU14O2 + HO2 = RU14OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 271] = KRO2HO2*0.770           

        # Reaction (272) RU12O2 + HO2 = RU12OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 272] = KRO2HO2*0.706           

        # Reaction (273) RU10O2 + HO2 = RU10OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 273] = KRO2HO2*0.625           

        # Reaction (274) NRN6O2 + HO2 = NRN6OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 274] = KRO2HO2*0.387         

        # Reaction (275) NRN9O2 + HO2 = NRN9OOH                                             
        chem["RC"].data.loc[:, :, :, ts, 275] = KRO2HO2*0.520         

        # Reaction (276) NRN12O2 + HO2 = NRN12OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 276] = KRO2HO2*0.625         

        # Reaction (277) NRU14O2 + HO2 = NRU14OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 277] = KRO2HO2*0.770         

        # Reaction (278) NRU12O2 + HO2 = NRU12OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 278] = KRO2HO2*0.625         

        # Reaction (279) RTN28O2 + HO2 = RTN28OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 279] = KRO2HO2*0.914         

        # Reaction (280) NRTN28O2 + HO2 = NRTN28OOH                                         
        chem["RC"].data.loc[:, :, :, ts, 280] = KRO2HO2*0.914         

        # Reaction (281) RTN26O2 + HO2 = RTN26OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 281] = KAPHO2                     

        # Reaction (282) RTN25O2 + HO2 = RTN25OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 282] = KRO2HO2*0.890       

        # Reaction (283) RTN24O2 + HO2 = RTN24OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 283] = KRO2HO2*0.890       

        # Reaction (284) RTN23O2 + HO2 = RTN23OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 284] = KRO2HO2*0.890       

        # Reaction (285) RTN14O2 + HO2 = RTN14OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 285] = KRO2HO2*0.770       

        # Reaction (286) RTN10O2 + HO2 = RTN10OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 286] = KRO2HO2*0.706       

        # Reaction (287) RTX28O2 + HO2 = RTX28OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 287] = KRO2HO2*0.914       

        # Reaction (288) RTX24O2 + HO2 = RTX24OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 288] = KRO2HO2*0.890       

        # Reaction (289) RTX22O2 + HO2 = RTX22OOH                                           
        chem["RC"].data.loc[:, :, :, ts, 289] = KRO2HO2*0.890       

        # Reaction (290) NRTX28O2 + HO2 = NRTX28OOH                                         
        chem["RC"].data.loc[:, :, :, ts, 290] = KRO2HO2*0.914       

        # Reaction (291) CH3O2 = HCHO + HO2                                                 
        chem["RC"].data.loc[:, :, :, ts, 291] = 1.82e-13*np.exp(416/temp)*0.33*RO2  

        # Reaction (292) CH3O2 = HCHO                                                       
        chem["RC"].data.loc[:, :, :, ts, 292] = 1.82e-13*np.exp(416/temp)*0.335*RO2 

        # Reaction (293) CH3O2 = CH3OH                                                      
        chem["RC"].data.loc[:, :, :, ts, 293] = 1.82e-13*np.exp(416/temp)*0.335*RO2 

        # Reaction (294) C2H5O2 = CH3CHO + HO2                                              
        chem["RC"].data.loc[:, :, :, ts, 294] = 3.10e-13*0.6*RO2             

        # Reaction (295) C2H5O2 = CH3CHO                                                    
        chem["RC"].data.loc[:, :, :, ts, 295] = 3.10e-13*0.2*RO2             

        # Reaction (296) C2H5O2 = C2H5OH                                                    
        chem["RC"].data.loc[:, :, :, ts, 296] = 3.10e-13*0.2*RO2             

        # Reaction (297) RN10O2 = C2H5CHO + HO2                                             
        chem["RC"].data.loc[:, :, :, ts, 297] = 6.00e-13*0.6*RO2             

        # Reaction (298) RN10O2 = C2H5CHO                                                   
        chem["RC"].data.loc[:, :, :, ts, 298] = 6.00e-13*0.2*RO2             

        # Reaction (299) RN10O2 = NPROPOL                                                   
        chem["RC"].data.loc[:, :, :, ts, 299] = 6.00e-13*0.2*RO2             

        # Reaction (300) IC3H7O2 = CH3COCH3 + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 300] = 4.00e-14*0.6*RO2             

        # Reaction (301) IC3H7O2 = CH3COCH3                                                 
        chem["RC"].data.loc[:, :, :, ts, 301] = 4.00e-14*0.2*RO2             

        # Reaction (302) IC3H7O2 = IPROPOL                                                  
        chem["RC"].data.loc[:, :, :, ts, 302] = 4.00e-14*0.2*RO2             

        # Reaction (303) RN13O2 = CH3CHO + C2H5O2                                           
        chem["RC"].data.loc[:, :, :, ts, 303] = 2.50e-13*RO2*BR01       

        # Reaction (304) RN13O2 = CARB11A + HO2                                             
        chem["RC"].data.loc[:, :, :, ts, 304] = 2.50e-13*RO2*(1-BR01)   

        # Reaction (305) RN13AO2 = RN12O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 305] = 8.80e-13*RO2                 

        # Reaction (306) RN16AO2 = RN15O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 306] = 8.80e-13*RO2                 

        # Reaction (307) RA13O2 = CARB3 + UDCARB8 + HO2                                     
        chem["RC"].data.loc[:, :, :, ts, 307] = 8.80e-13*RO2                 

        # Reaction (308) RA16O2 = CARB3 + UDCARB11 + HO2                                    
        chem["RC"].data.loc[:, :, :, ts, 308] = 8.80e-13*RO2*0.7          

        # Reaction (309) RA16O2 = CARB6 + UDCARB8 + HO2                                     
        chem["RC"].data.loc[:, :, :, ts, 309] = 8.80e-13*RO2*0.3          

        # Reaction (310) RA19AO2 = CARB3 + UDCARB14 + HO2                                   
        chem["RC"].data.loc[:, :, :, ts, 310] = 8.80e-13*RO2                 

        # Reaction (311) RA19CO2 = CARB3 + UDCARB14 + HO2                                   
        chem["RC"].data.loc[:, :, :, ts, 311] = 8.80e-13*RO2                 

        # Reaction (312) RN16O2 = RN15AO2                                                   
        chem["RC"].data.loc[:, :, :, ts, 312] = 2.50e-13*RO2                 

        # Reaction (313) RN19O2 = RN18AO2                                                   
        chem["RC"].data.loc[:, :, :, ts, 313] = 2.50e-13*RO2                 

        # Reaction (314) HOCH2CH2O2 = HCHO + HCHO + HO2                                     
        chem["RC"].data.loc[:, :, :, ts, 314] = 2.00e-12*RO2*0.776       

        # Reaction (315) HOCH2CH2O2 = HOCH2CHO + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 315] = 2.00e-12*RO2*0.224       

        # Reaction (316) RN9O2 = CH3CHO + HCHO + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 316] = 8.80e-13*RO2                 

        # Reaction (317) RN12O2 = CH3CHO + CH3CHO + HO2                                     
        chem["RC"].data.loc[:, :, :, ts, 317] = 8.80e-13*RO2                 

        # Reaction (318) RN15O2 = C2H5CHO + CH3CHO + HO2                                    
        chem["RC"].data.loc[:, :, :, ts, 318] = 8.80e-13*RO2                 

        # Reaction (319) RN18O2 = C2H5CHO + C2H5CHO + HO2                                   
        chem["RC"].data.loc[:, :, :, ts, 319] = 8.80e-13*RO2                 

        # Reaction (320) RN15AO2 = CARB13 + HO2                                             
        chem["RC"].data.loc[:, :, :, ts, 320] = 8.80e-13*RO2                 

        # Reaction (321) RN18AO2 = CARB16 + HO2                                             
        chem["RC"].data.loc[:, :, :, ts, 321] = 8.80e-13*RO2                 

        # Reaction (322) CH3CO3 = CH3O2                                                     
        chem["RC"].data.loc[:, :, :, ts, 322] = 1.00e-11*RO2                 

        # Reaction (323) C2H5CO3 = C2H5O2                                                   
        chem["RC"].data.loc[:, :, :, ts, 323] = 1.00e-11*RO2                 

        # Reaction (324) HOCH2CO3 = HCHO + HO2                                              
        chem["RC"].data.loc[:, :, :, ts, 324] = 1.00e-11*RO2                 

        # Reaction (325) RN8O2 = CH3CO3 + HCHO                                              
        chem["RC"].data.loc[:, :, :, ts, 325] = 1.40e-12*RO2                 

        # Reaction (326) RN11O2 = CH3CO3 + CH3CHO                                           
        chem["RC"].data.loc[:, :, :, ts, 326] = 1.40e-12*RO2                 

        # Reaction (327) RN14O2 = C2H5CO3 + CH3CHO                                          
        chem["RC"].data.loc[:, :, :, ts, 327] = 1.40e-12*RO2                 

        # Reaction (328) RN17O2 = RN16AO2                                                   
        chem["RC"].data.loc[:, :, :, ts, 328] = 1.40e-12*RO2                 

        # Reaction (329) RU14O2 = UCARB12 + HO2                                             
        chem["RC"].data.loc[:, :, :, ts, 329] = 1.71e-12*RO2*0.252        

        # Reaction (330) RU14O2 = UCARB10 + HCHO + HO2                                      
        chem["RC"].data.loc[:, :, :, ts, 330] = 1.71e-12*RO2*0.748        

        # Reaction (331) RU12O2 = CH3CO3 + HOCH2CHO                                         
        chem["RC"].data.loc[:, :, :, ts, 331] = 2.00e-12*RO2*0.7            

        # Reaction (332) RU12O2 = CARB7 + HOCH2CHO + HO2                                    
        chem["RC"].data.loc[:, :, :, ts, 332] = 2.00e-12*RO2*0.3            

        # Reaction (333) RU10O2 = CH3CO3 + HOCH2CHO                                         
        chem["RC"].data.loc[:, :, :, ts, 333] = 2.00e-12*RO2*0.5            

        # Reaction (334) RU10O2 = CARB6 + HCHO + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 334] = 2.00e-12*RO2*0.3            

        # Reaction (335) RU10O2 = CARB7 + HCHO + HO2                                        
        chem["RC"].data.loc[:, :, :, ts, 335] = 2.00e-12*RO2*0.2            

        # Reaction (336) NRN6O2 = HCHO + HCHO + NO2                                         
        chem["RC"].data.loc[:, :, :, ts, 336] = 6.00e-13*RO2                 

        # Reaction (337) NRN9O2 = CH3CHO + HCHO + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 337] = 2.30e-13*RO2                 

        # Reaction (338) NRN12O2 = CH3CHO + CH3CHO + NO2                                    
        chem["RC"].data.loc[:, :, :, ts, 338] = 2.50e-13*RO2                 

        # Reaction (339) NRU14O2 = NUCARB12 + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 339] = 1.30e-12*RO2                 

        # Reaction (340) NRU12O2 = NOA + CO + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 340] = 9.60e-13*RO2                 

        # Reaction (341) RTN28O2 = TNCARB26 + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 341] = 2.85e-13*RO2                 

        # Reaction (342) NRTN28O2 = TNCARB26 + NO2                                          
        chem["RC"].data.loc[:, :, :, ts, 342] = 1.00e-13*RO2                 

        # Reaction (343) RTN26O2 = RTN25O2                                                  
        chem["RC"].data.loc[:, :, :, ts, 343] = 1.00e-11*RO2                   

        # Reaction (344) RTN25O2 = RTN24O2                                                  
        chem["RC"].data.loc[:, :, :, ts, 344] = 1.30e-12*RO2           

        # Reaction (345) RTN24O2 = RTN23O2                                                  
        chem["RC"].data.loc[:, :, :, ts, 345] = 6.70e-15*RO2             

        # Reaction (346) RTN23O2 = CH3COCH3 + RTN14O2                                       
        chem["RC"].data.loc[:, :, :, ts, 346] = 6.70e-15*RO2            

        # Reaction (347) RTN14O2 = HCHO + TNCARB10 + HO2                                    
        chem["RC"].data.loc[:, :, :, ts, 347] = 8.80e-13*RO2        

        # Reaction (348) RTN10O2 = RN8O2 + CO                                               
        chem["RC"].data.loc[:, :, :, ts, 348] = 2.00e-12*RO2        

        # Reaction (349) RTX28O2 = TXCARB24 + HCHO + HO2                                    
        chem["RC"].data.loc[:, :, :, ts, 349] = 2.00e-12*RO2       

        # Reaction (350) RTX24O2 = TXCARB22 + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 350] = 2.50e-13*RO2       

        # Reaction (351) RTX22O2 = CH3COCH3 + RN13O2                                        
        chem["RC"].data.loc[:, :, :, ts, 351] = 2.50e-13*RO2       

        # Reaction (352) NRTX28O2 = TXCARB24 + HCHO + NO2                                   
        chem["RC"].data.loc[:, :, :, ts, 352] = 9.20e-14*RO2       

        # Reaction (353) OH + CARB14 = RN14O2                                               
        chem["RC"].data.loc[:, :, :, ts, 353] = 1.87e-11       

        # Reaction (354) OH + CARB17 = RN17O2                                               
        chem["RC"].data.loc[:, :, :, ts, 354] = 4.36e-12       

        # Reaction (355) OH + CARB11A = RN11O2                                              
        chem["RC"].data.loc[:, :, :, ts, 355] = 3.24e-18*temp**2*np.exp(414/temp)

        # Reaction (356) OH + CARB7 = CARB6 + HO2                                           
        chem["RC"].data.loc[:, :, :, ts, 356] = 3.00e-12       

        # Reaction (357) OH + CARB10 = CARB9 + HO2                                          
        chem["RC"].data.loc[:, :, :, ts, 357] = 5.86e-12       

        # Reaction (358) OH + CARB13 = RN13O2                                               
        chem["RC"].data.loc[:, :, :, ts, 358] = 1.65e-11       

        # Reaction (359) OH + CARB16 = RN16O2                                               
        chem["RC"].data.loc[:, :, :, ts, 359] = 1.25e-11       

        # Reaction (360) OH + UCARB10 = RU10O2                                              
        chem["RC"].data.loc[:, :, :, ts, 360] = 2.50e-11       

        # Reaction (361) NO3 + UCARB10 = RU10O2 + HNO3                                      
        chem["RC"].data.loc[:, :, :, ts, 361] = KNO3AL       

        # Reaction (362) O3 + UCARB10 = HCHO + CH3CO3 + CO + OH                             
        chem["RC"].data.loc[:, :, :, ts, 362] = 2.85e-18*0.59       

        # Reaction (363) O3 + UCARB10 = HCHO + CARB6 + H2O2                                 
        chem["RC"].data.loc[:, :, :, ts, 363] = 2.85e-18*0.41       

        # Reaction (364) OH + HOCH2CHO = HOCH2CO3                                           
        chem["RC"].data.loc[:, :, :, ts, 364] = 1.00e-11       

        # Reaction (365) NO3 + HOCH2CHO = HOCH2CO3 + HNO3                                   
        chem["RC"].data.loc[:, :, :, ts, 365] = KNO3AL        

        # Reaction (366) OH + CARB3 = CO + CO + HO2                                         
        chem["RC"].data.loc[:, :, :, ts, 366] = 1.14e-11       

        # Reaction (367) OH + CARB6 = CH3CO3 + CO                                           
        chem["RC"].data.loc[:, :, :, ts, 367] = 1.72e-11       

        # Reaction (368) OH + CARB9 = RN9O2                                                 
        chem["RC"].data.loc[:, :, :, ts, 368] = 2.40e-13       

        # Reaction (369) OH + CARB12 = RN12O2                                               
        chem["RC"].data.loc[:, :, :, ts, 369] = 1.38e-12       

        # Reaction (370) OH + CARB15 = RN15O2                                               
        chem["RC"].data.loc[:, :, :, ts, 370] = 4.81e-12       

        # Reaction (371) OH + CCARB12 = RN12O2                                              
        chem["RC"].data.loc[:, :, :, ts, 371] = 4.79e-12       

        # Reaction (372) OH + UCARB12 = RU12O2                                              
        chem["RC"].data.loc[:, :, :, ts, 372] = 4.52e-11            

        # Reaction (373) NO3 + UCARB12 = RU12O2 + HNO3                                      
        chem["RC"].data.loc[:, :, :, ts, 373] = KNO3AL*4.25    

        # Reaction (374) O3 + UCARB12 = HOCH2CHO + CH3CO3 + CO + OH                         
        chem["RC"].data.loc[:, :, :, ts, 374] = 2.40e-17*0.89   

        # Reaction (375) O3 + UCARB12 = HOCH2CHO + CARB6 + H2O2                             
        chem["RC"].data.loc[:, :, :, ts, 375] = 2.40e-17*0.11   

        # Reaction (376) OH + NUCARB12 = NRU12O2                                            
        chem["RC"].data.loc[:, :, :, ts, 376] = 4.16e-11            

        # Reaction (377) OH + NOA = CARB6 + NO2                                             
        chem["RC"].data.loc[:, :, :, ts, 377] = 1.30e-13            

        # Reaction (378) OH + UDCARB8 = C2H5O2                                              
        chem["RC"].data.loc[:, :, :, ts, 378] = 5.20e-11*0.50        

        # Reaction (379) OH + UDCARB8 = ANHY + HO2                                          
        chem["RC"].data.loc[:, :, :, ts, 379] = 5.20e-11*0.50        

        # Reaction (380) OH + UDCARB11 = RN10O2                                             
        chem["RC"].data.loc[:, :, :, ts, 380] = 5.58e-11*0.55     

        # Reaction (381) OH + UDCARB11 = ANHY + CH3O2                                       
        chem["RC"].data.loc[:, :, :, ts, 381] = 5.58e-11*0.45     

        # Reaction (382) OH + UDCARB14 = RN13O2                                             
        chem["RC"].data.loc[:, :, :, ts, 382] = 7.00e-11*0.55     

        # Reaction (383) OH + UDCARB14 = ANHY + C2H5O2                                      
        chem["RC"].data.loc[:, :, :, ts, 383] = 7.00e-11*0.45     

        # Reaction (384) OH + TNCARB26 = RTN26O2                                            
        chem["RC"].data.loc[:, :, :, ts, 384] = 4.20e-11           

        # Reaction (385) OH + TNCARB15 = RN15AO2                                            
        chem["RC"].data.loc[:, :, :, ts, 385] = 1.00e-12           

        # Reaction (386) OH + TNCARB10 = RTN10O2                                            
        chem["RC"].data.loc[:, :, :, ts, 386] = 1.00e-10           

        # Reaction (387) NO3 + TNCARB26 = RTN26O2 + HNO3                                    
        chem["RC"].data.loc[:, :, :, ts, 387] = 3.80e-14            

        # Reaction (388) NO3 + TNCARB10 = RTN10O2 + HNO3                                    
        chem["RC"].data.loc[:, :, :, ts, 388] = KNO3AL*5.5      

        # Reaction (389) OH + RCOOH25 = RTN25O2                                             
        chem["RC"].data.loc[:, :, :, ts, 389] = 6.65e-12            

        # Reaction (390) OH + TXCARB24 = RTX24O2                                            
        chem["RC"].data.loc[:, :, :, ts, 390] = 1.55e-11           

        # Reaction (391) OH + TXCARB22 = RTX22O2                                            
        chem["RC"].data.loc[:, :, :, ts, 391] = 4.55e-12           

        # Reaction (392) OH + CH3NO3 = HCHO + NO2                                           
        chem["RC"].data.loc[:, :, :, ts, 392] = 1.00e-14*np.exp(1060/temp)      

        # Reaction (393) OH + C2H5NO3 = CH3CHO + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 393] = 4.40e-14*np.exp(720/temp)       

        # Reaction (394) OH + RN10NO3 = C2H5CHO + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 394] = 7.30e-13                     

        # Reaction (395) OH + IC3H7NO3 = CH3COCH3 + NO2                                     
        chem["RC"].data.loc[:, :, :, ts, 395] = 4.90e-13                     

        # Reaction (396) OH + RN13NO3 = CARB11A + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 396] = 9.20e-13                     

        # Reaction (397) OH + RN16NO3 = CARB14 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 397] = 1.85e-12                     

        # Reaction (398) OH + RN19NO3 = CARB17 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 398] = 3.02e-12                     

        # Reaction (399) OH + HOC2H4NO3 = HOCH2CHO + NO2                                    
        chem["RC"].data.loc[:, :, :, ts, 399] = 1.09e-12               

        # Reaction (400) OH + RN9NO3 = CARB7 + NO2                                          
        chem["RC"].data.loc[:, :, :, ts, 400] = 1.31e-12               

        # Reaction (401) OH + RN12NO3 = CARB10 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 401] = 1.79e-12               

        # Reaction (402) OH + RN15NO3 = CARB13 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 402] = 1.03e-11               

        # Reaction (403) OH + RN18NO3 = CARB16 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 403] = 1.34e-11               

        # Reaction (404) OH + RU14NO3 = UCARB12 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 404] = 5.55e-11               

        # Reaction (405) OH + RA13NO3 = CARB3 + UDCARB8 + NO2                               
        chem["RC"].data.loc[:, :, :, ts, 405] = 7.30e-11               

        # Reaction (406) OH + RA16NO3 = CARB3 + UDCARB11 + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 406] = 7.16e-11               

        # Reaction (407) OH + RA19NO3 = CARB6 + UDCARB11 + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 407] = 8.31e-11               

        # Reaction (408) OH + RTN28NO3 = TNCARB26 + NO2                                     
        chem["RC"].data.loc[:, :, :, ts, 408] = 4.35e-12               

        # Reaction (409) OH + RTN25NO3 = CH3COCH3 + TNCARB15 + NO2                          
        chem["RC"].data.loc[:, :, :, ts, 409] = 2.88e-12               

        # Reaction (410) OH + RTX28NO3 = TXCARB24 + HCHO + NO2                              
        chem["RC"].data.loc[:, :, :, ts, 410] = 3.53e-12                  

        # Reaction (411) OH + RTX24NO3 = TXCARB22 + NO2                                     
        chem["RC"].data.loc[:, :, :, ts, 411] = 6.48e-12                  

        # Reaction (412) OH + RTX22NO3 = CH3COCH3 + CCARB12 + NO2                           
        chem["RC"].data.loc[:, :, :, ts, 412] = 4.74e-12                  

        # Reaction (413) OH + AROH14 = RAROH14                                              
        chem["RC"].data.loc[:, :, :, ts, 413] = 2.63e-11             

        # Reaction (414) NO3 + AROH14 = RAROH14 + HNO3                                      
        chem["RC"].data.loc[:, :, :, ts, 414] = 3.78e-12               

        # Reaction (415) RAROH14 + NO2 = ARNOH14                                            
        chem["RC"].data.loc[:, :, :, ts, 415] = 2.08e-12               

        # Reaction (416) OH + ARNOH14 = CARB13 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 416] = 9.00e-13               

        # Reaction (417) NO3 + ARNOH14 = CARB13 + NO2 + HNO3                                
        chem["RC"].data.loc[:, :, :, ts, 417] = 9.00e-14               

        # Reaction (418) OH + AROH17 = RAROH17                                              
        chem["RC"].data.loc[:, :, :, ts, 418] = 4.65e-11               

        # Reaction (419) NO3 + AROH17 = RAROH17 + HNO3                                      
        chem["RC"].data.loc[:, :, :, ts, 419] = 1.25e-11               

        # Reaction (420) RAROH17 + NO2 = ARNOH17                                            
        chem["RC"].data.loc[:, :, :, ts, 420] = 2.08e-12               

        # Reaction (421) OH + ARNOH17 = CARB16 + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 421] = 1.53e-12               

        # Reaction (422) NO3 + ARNOH17 = CARB16 + NO2 + HNO3                                
        chem["RC"].data.loc[:, :, :, ts, 422] = 3.13e-13               

        # Reaction (423) OH + CH3OOH = CH3O2                                                
        chem["RC"].data.loc[:, :, :, ts, 423] = 1.90e-11*np.exp(190/temp)       

        # Reaction (424) OH + CH3OOH = HCHO + OH                                            
        chem["RC"].data.loc[:, :, :, ts, 424] = 1.00e-11*np.exp(190/temp)       

        # Reaction (425) OH + C2H5OOH = CH3CHO + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 425] = 1.36e-11               

        # Reaction (426) OH + RN10OOH = C2H5CHO + OH                                        
        chem["RC"].data.loc[:, :, :, ts, 426] = 1.89e-11               

        # Reaction (427) OH + IC3H7OOH = CH3COCH3 + OH                                      
        chem["RC"].data.loc[:, :, :, ts, 427] = 2.78e-11               

        # Reaction (428) OH + RN13OOH = CARB11A + OH                                        
        chem["RC"].data.loc[:, :, :, ts, 428] = 3.57e-11               

        # Reaction (429) OH + RN16OOH = CARB14 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 429] = 4.21e-11               

        # Reaction (430) OH + RN19OOH = CARB17 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 430] = 4.71e-11               

        # Reaction (431) OH + CH3CO3H = CH3CO3                                              
        chem["RC"].data.loc[:, :, :, ts, 431] = 3.70e-12                     

        # Reaction (432) OH + C2H5CO3H = C2H5CO3                                            
        chem["RC"].data.loc[:, :, :, ts, 432] = 4.42e-12                     

        # Reaction (433) OH + HOCH2CO3H = HOCH2CO3                                          
        chem["RC"].data.loc[:, :, :, ts, 433] = 6.19e-12                     

        # Reaction (434) OH + RN8OOH = CARB6 + OH                                           
        chem["RC"].data.loc[:, :, :, ts, 434] = 4.42e-12                     

        # Reaction (435) OH + RN11OOH = CARB9 + OH                                          
        chem["RC"].data.loc[:, :, :, ts, 435] = 2.50e-11                     

        # Reaction (436) OH + RN14OOH = CARB12 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 436] = 3.20e-11                     

        # Reaction (437) OH + RN17OOH = CARB15 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 437] = 3.35e-11                     

        # Reaction (438) OH + RU14OOH = UCARB12 + OH                                        
        chem["RC"].data.loc[:, :, :, ts, 438] = 7.51e-11                     

        # Reaction (439) OH + RU12OOH = RU12O2                                              
        chem["RC"].data.loc[:, :, :, ts, 439] = 3.00e-11                     

        # Reaction (440) OH + RU10OOH = RU10O2                                              
        chem["RC"].data.loc[:, :, :, ts, 440] = 3.00e-11                     

        # Reaction (441) OH + NRU14OOH = NUCARB12 + OH                                      
        chem["RC"].data.loc[:, :, :, ts, 441] = 1.03e-10                     

        # Reaction (442) OH + NRU12OOH = NOA + CO + OH                                      
        chem["RC"].data.loc[:, :, :, ts, 442] = 2.65e-11                     

        # Reaction (443) OH + HOC2H4OOH = HOCH2CHO + OH                                     
        chem["RC"].data.loc[:, :, :, ts, 443] = 2.13e-11               

        # Reaction (444) OH + RN9OOH = CARB7 + OH                                           
        chem["RC"].data.loc[:, :, :, ts, 444] = 2.50e-11               

        # Reaction (445) OH + RN12OOH = CARB10 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 445] = 3.25e-11               

        # Reaction (446) OH + RN15OOH = CARB13 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 446] = 3.74e-11               

        # Reaction (447) OH + RN18OOH = CARB16 + OH                                         
        chem["RC"].data.loc[:, :, :, ts, 447] = 3.83e-11               

        # Reaction (448) OH + NRN6OOH = HCHO + HCHO + NO2 + OH                              
        chem["RC"].data.loc[:, :, :, ts, 448] = 5.22e-12               

        # Reaction (449) OH + NRN9OOH = CH3CHO + HCHO + NO2 + OH                            
        chem["RC"].data.loc[:, :, :, ts, 449] = 6.50e-12               

        # Reaction (450) OH + NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                         
        chem["RC"].data.loc[:, :, :, ts, 450] = 7.15e-12               

        # Reaction (451) OH + RA13OOH = CARB3 + UDCARB8 + OH                                
        chem["RC"].data.loc[:, :, :, ts, 451] = 9.77e-11               

        # Reaction (452) OH + RA16OOH = CARB3 + UDCARB11 + OH                               
        chem["RC"].data.loc[:, :, :, ts, 452] = 9.64e-11               

        # Reaction (453) OH + RA19OOH = CARB6 + UDCARB11 + OH                               
        chem["RC"].data.loc[:, :, :, ts, 453] = 1.12e-10               

        # Reaction (454) OH + RTN28OOH = TNCARB26 + OH                                      
        chem["RC"].data.loc[:, :, :, ts, 454] = 2.38e-11               

        # Reaction (455) OH + RTN26OOH = RTN26O2                                            
        chem["RC"].data.loc[:, :, :, ts, 455] = 1.20e-11               

        # Reaction (456) OH + NRTN28OOH = TNCARB26 + NO2 + OH                               
        chem["RC"].data.loc[:, :, :, ts, 456] = 9.50e-12               

        # Reaction (457) OH + RTN25OOH = RTN25O2                                            
        chem["RC"].data.loc[:, :, :, ts, 457] = 1.66e-11               

        # Reaction (458) OH + RTN24OOH = RTN24O2                                            
        chem["RC"].data.loc[:, :, :, ts, 458] = 1.05e-11               

        # Reaction (459) OH + RTN23OOH = RTN23O2                                            
        chem["RC"].data.loc[:, :, :, ts, 459] = 2.05e-11               

        # Reaction (460) OH + RTN14OOH = RTN14O2                                            
        chem["RC"].data.loc[:, :, :, ts, 460] = 8.69e-11               

        # Reaction (461) OH + RTN10OOH = RTN10O2                                            
        chem["RC"].data.loc[:, :, :, ts, 461] = 4.23e-12               

        # Reaction (462) OH + RTX28OOH = RTX28O2                                            
        chem["RC"].data.loc[:, :, :, ts, 462] = 2.00e-11               

        # Reaction (463) OH + RTX24OOH = TXCARB22 + OH                                      
        chem["RC"].data.loc[:, :, :, ts, 463] = 8.59e-11               

        # Reaction (464) OH + RTX22OOH = CH3COCH3 + CCARB12 + OH                            
        chem["RC"].data.loc[:, :, :, ts, 464] = 7.50e-11               

        # Reaction (465) OH + NRTX28OOH = NRTX28O2                                          
        chem["RC"].data.loc[:, :, :, ts, 465] = 9.58e-12               

        # Reaction (466) OH + ANHY = HOCH2CH2O2                                             
        chem["RC"].data.loc[:, :, :, ts, 466] = 1.50e-12        

        # Reaction (467) CH3CO3 + NO2 = PAN                                                 
        chem["RC"].data.loc[:, :, :, ts, 467] = KFPAN                        

        # Reaction (468) PAN = CH3CO3 + NO2                                                 
        chem["RC"].data.loc[:, :, :, ts, 468] = KBPAN                        

        # Reaction (469) C2H5CO3 + NO2 = PPN                                                
        chem["RC"].data.loc[:, :, :, ts, 469] = KFPAN                        

        # Reaction (470) PPN = C2H5CO3 + NO2                                                
        chem["RC"].data.loc[:, :, :, ts, 470] = KBPAN                        

        # Reaction (471) HOCH2CO3 + NO2 = PHAN                                              
        chem["RC"].data.loc[:, :, :, ts, 471] = KFPAN                        

        # Reaction (472) PHAN = HOCH2CO3 + NO2                                              
        chem["RC"].data.loc[:, :, :, ts, 472] = KBPAN                        

        # Reaction (473) OH + PAN = HCHO + CO + NO2                                         
        chem["RC"].data.loc[:, :, :, ts, 473] = 9.50e-13*np.exp(-650/temp)      

        # Reaction (474) OH + PPN = CH3CHO + CO + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 474] = 1.27e-12                       

        # Reaction (475) OH + PHAN = HCHO + CO + NO2                                        
        chem["RC"].data.loc[:, :, :, ts, 475] = 1.12e-12                       

        # Reaction (476) RU12O2 + NO2 = RU12PAN                                             
        chem["RC"].data.loc[:, :, :, ts, 476] = KFPAN*0.061             

        # Reaction (477) RU12PAN = RU12O2 + NO2                                             
        chem["RC"].data.loc[:, :, :, ts, 477] = KBPAN                   

        # Reaction (478) RU10O2 + NO2 = MPAN                                                
        chem["RC"].data.loc[:, :, :, ts, 478] = KFPAN*0.041             

        # Reaction (479) MPAN = RU10O2 + NO2                                                
        chem["RC"].data.loc[:, :, :, ts, 479] = KBPAN                  

        # Reaction (480) OH + MPAN = CARB7 + CO + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 480] = 3.60e-12 

        # Reaction (481) OH + RU12PAN = UCARB10 + NO2                                       
        chem["RC"].data.loc[:, :, :, ts, 481] = 2.52e-11 

        # Reaction (482) RTN26O2 + NO2 = RTN26PAN                                           
        chem["RC"].data.loc[:, :, :, ts, 482] = KFPAN*0.722      

        # Reaction (483) RTN26PAN = RTN26O2 + NO2                                           
        chem["RC"].data.loc[:, :, :, ts, 483] = KBPAN                   

        # Reaction (484) OH + RTN26PAN = CH3COCH3 + CARB16 + NO2                            
        chem["RC"].data.loc[:, :, :, ts, 484] = 3.66e-12  

        # Reaction (485) RTN28NO3 = P2604                                                   
        chem["RC"].data.loc[:, :, :, ts, 485] = KIN    

        # Reaction (486) P2604 = RTN28NO3                                                   
        chem["RC"].data.loc[:, :, :, ts, 486] = KOUT2604 

        # Reaction (487) RTX28NO3 = P4608                                                   
        chem["RC"].data.loc[:, :, :, ts, 487] = KIN 

        # Reaction (488) P4608 = RTX28NO3                                                   
        chem["RC"].data.loc[:, :, :, ts, 488] = KOUT4608 

        # Reaction (489) RCOOH25 = P2631                                                    
        chem["RC"].data.loc[:, :, :, ts, 489] = KIN  

        # Reaction (490) P2631 = RCOOH25                                                    
        chem["RC"].data.loc[:, :, :, ts, 490] = KOUT2631 

        # Reaction (491) RTN24OOH = P2635                                                   
        chem["RC"].data.loc[:, :, :, ts, 491] = KIN  

        # Reaction (492) P2635 = RTN24OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 492] = KOUT2635 

        # Reaction (493) RTX28OOH = P4610                                                   
        chem["RC"].data.loc[:, :, :, ts, 493] = KIN  

        # Reaction (494) P4610 = RTX28OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 494] = KOUT4610 

        # Reaction (495) RTN28OOH = P2605                                                   
        chem["RC"].data.loc[:, :, :, ts, 495] = KIN  

        # Reaction (496) P2605 = RTN28OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 496] = KOUT2605 

        # Reaction (497) RTN26OOH = P2630                                                   
        chem["RC"].data.loc[:, :, :, ts, 497] = KIN  

        # Reaction (498) P2630 = RTN26OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 498] = KOUT2630 

        # Reaction (499) RTN26PAN = P2629                                                   
        chem["RC"].data.loc[:, :, :, ts, 499] = KIN  

        # Reaction (500) P2629 = RTN26PAN                                                   
        chem["RC"].data.loc[:, :, :, ts, 500] = KOUT2629 

        # Reaction (501) RTN25OOH = P2632                                                   
        chem["RC"].data.loc[:, :, :, ts, 501] = KIN 

        # Reaction (502) P2632 = RTN25OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 502] = KOUT2632 

        # Reaction (503) RTN23OOH = P2637                                                   
        chem["RC"].data.loc[:, :, :, ts, 503] = KIN  

        # Reaction (504) P2637 = RTN23OOH                                                   
        chem["RC"].data.loc[:, :, :, ts, 504] = KOUT2637 

        # Reaction (505) ARNOH14 = P3612                                                    
        chem["RC"].data.loc[:, :, :, ts, 505] = KIN  

        # Reaction (506) P3612 = ARNOH14                                                    
        chem["RC"].data.loc[:, :, :, ts, 506] = KOUT3612 

        # Reaction (507) ARNOH17 = P3613                                                    
        chem["RC"].data.loc[:, :, :, ts, 507] = KIN 

        # Reaction (508) P3613 = ARNOH17                                                    
        chem["RC"].data.loc[:, :, :, ts, 508] = KOUT3613 

        # Reaction (509) ANHY = P3442                                                       
        chem["RC"].data.loc[:, :, :, ts, 509] = KIN  

        # Reaction (510) P3442 = ANHY                                                       
        chem["RC"].data.loc[:, :, :, ts, 510] = KOUT3442 


    def _photol(self, ts):
        """Calculate photolysis rates for each grid cell and timestep"""

        met = self.met
        chem = self.chem

        J = chem["J"].data.expand_dims(dim={'level': chem.data["level"]}, axis=2)
        BR01 = chem["BR01"].data.sel(time=ts)

        # Photol Reaction (1) O3 = O1D
        chem["DJ"].data.loc[:, :, :, ts, 1] = J.loc[:, :, :, ts, 1]       

        # Photol Reaction (2) O3 = O                                                             
        chem["DJ"].data.loc[:, :, :, ts, 2] = J.loc[:, :, :, ts, 2]                             
         
        # Photol Reaction (3) H2O2 = OH + OH                                                     
        chem["DJ"].data.loc[:, :, :, ts, 3] = J.loc[:, :, :, ts, 3]                             
         
        # Photol Reaction (4) NO2 = NO + O                                                       
        chem["DJ"].data.loc[:, :, :, ts, 4] = J.loc[:, :, :, ts, 4]                             
         
        # Photol Reaction (5) NO3 = NO                                                           
        chem["DJ"].data.loc[:, :, :, ts, 5] = J.loc[:, :, :, ts, 5]                             
         
        # Photol Reaction (6) NO3 = NO2 + O                                                      
        chem["DJ"].data.loc[:, :, :, ts, 6] = J.loc[:, :, :, ts, 6]                             
         
        # Photol Reaction (7) HONO = OH + NO                                                     
        chem["DJ"].data.loc[:, :, :, ts, 7] = J.loc[:, :, :, ts, 7]                             
         
        # Photol Reaction (8) HNO3 = OH + NO2                                                    
        chem["DJ"].data.loc[:, :, :, ts, 8] = J.loc[:, :, :, ts, 8]                             
         
        # Photol Reaction (9) HCHO = CO + HO2 + HO2                                              
        chem["DJ"].data.loc[:, :, :, ts, 9] = J.loc[:, :, :, ts, 11]                        
         
        # Photol Reaction (10) HCHO = H2 + CO                                                     
        chem["DJ"].data.loc[:, :, :, ts, 10] = J.loc[:, :, :, ts, 12]                        
         
        # Photol Reaction (11) CH3CHO = CH3O2 + HO2 + CO                                          
        chem["DJ"].data.loc[:, :, :, ts, 11] = J.loc[:, :, :, ts, 13]                        
         
        # Photol Reaction (12) C2H5CHO = C2H5O2 + CO + HO2                                        
        chem["DJ"].data.loc[:, :, :, ts, 12] = J.loc[:, :, :, ts, 14]                        
         
        # Photol Reaction (13) CH3COCH3 = CH3CO3 + CH3O2                                          
        chem["DJ"].data.loc[:, :, :, ts, 13] = J.loc[:, :, :, ts, 21]                        
         
        # Photol Reaction (14) MEK = CH3CO3 + C2H5O2                                              
        chem["DJ"].data.loc[:, :, :, ts, 14] = J.loc[:, :, :, ts, 22]                        
         
        # Photol Reaction (15) CARB14 = CH3CO3 + RN10O2                                           
        chem["DJ"].data.loc[:, :, :, ts, 15] = J.loc[:, :, :, ts, 22]*4.74               
         
        # Photol Reaction (16) CARB17 = RN8O2 + RN10O2                                            
        chem["DJ"].data.loc[:, :, :, ts, 16] = J.loc[:, :, :, ts, 22]*1.33               
         
        # Photol Reaction (17) CARB11A = CH3CO3 + C2H5O2                                          
        chem["DJ"].data.loc[:, :, :, ts, 17] = J.loc[:, :, :, ts, 22]                        
         
        # Photol Reaction (18) CARB7 = CH3CO3 + HCHO + HO2                                        
        chem["DJ"].data.loc[:, :, :, ts, 18] = J.loc[:, :, :, ts, 22]                        
         
        # Photol Reaction (19) CARB10 = CH3CO3 + CH3CHO + HO2                                     
        chem["DJ"].data.loc[:, :, :, ts, 19] = J.loc[:, :, :, ts, 22]                        
         
        # Photol Reaction (20) CARB13 = RN8O2 + CH3CHO + HO2                                      
        chem["DJ"].data.loc[:, :, :, ts, 20] = J.loc[:, :, :, ts, 22]*3.00               
         
        # Photol Reaction (21) CARB16 = RN8O2 + C2H5CHO + HO2                                     
        chem["DJ"].data.loc[:, :, :, ts, 21] = J.loc[:, :, :, ts, 22]*3.35               
         
        # Photol Reaction (22) HOCH2CHO = HCHO + CO + HO2 + HO2                                   
        chem["DJ"].data.loc[:, :, :, ts, 22] = J.loc[:, :, :, ts, 15]                        
         
        # Photol Reaction (23) UCARB10 = CH3CO3 + HCHO + HO2                                      
        chem["DJ"].data.loc[:, :, :, ts, 23] = J.loc[:, :, :, ts, 18]*2                       
         
        # Photol Reaction (24) CARB3 = CO + CO + HO2 + HO2                                        
        chem["DJ"].data.loc[:, :, :, ts, 24] = J.loc[:, :, :, ts, 33]                        
         
        # Photol Reaction (25) CARB6 = CH3CO3 + CO + HO2                                          
        chem["DJ"].data.loc[:, :, :, ts, 25] = J.loc[:, :, :, ts, 34]                        
         
        # Photol Reaction (26) CARB9 = CH3CO3 + CH3CO3                                            
        chem["DJ"].data.loc[:, :, :, ts, 26] = J.loc[:, :, :, ts, 35]                        
         
        # Photol Reaction (27) CARB12 = CH3CO3 + RN8O2                                            
        chem["DJ"].data.loc[:, :, :, ts, 27] = J.loc[:, :, :, ts, 35]                        
         
        # Photol Reaction (28) CARB15 = RN8O2 + RN8O2                                             
        chem["DJ"].data.loc[:, :, :, ts, 28] = J.loc[:, :, :, ts, 35]                        
         
        # Photol Reaction (29) UCARB12 = CH3CO3 + HOCH2CHO + CO + HO2                             
        chem["DJ"].data.loc[:, :, :, ts, 29] = J.loc[:, :, :, ts, 18]*2           
         
        # Photol Reaction (30) NUCARB12 = NOA + CO + CO + HO2 + HO2                               
        chem["DJ"].data.loc[:, :, :, ts, 30] = J.loc[:, :, :, ts, 18]             
         
        # Photol Reaction (31) NOA = CH3CO3 + HCHO + NO2                                          
        chem["DJ"].data.loc[:, :, :, ts, 31] = J.loc[:, :, :, ts, 56]             
         
        # Photol Reaction (32) NOA = CH3CO3 + HCHO + NO2                                          
        chem["DJ"].data.loc[:, :, :, ts, 32] = J.loc[:, :, :, ts, 57]             
         
        # Photol Reaction (33) UDCARB8 = C2H5O2 + HO2                                             
        chem["DJ"].data.loc[:, :, :, ts, 33] = J.loc[:, :, :, ts, 4]*0.02*0.64   
         
        # Photol Reaction (34) UDCARB8 = ANHY + HO2 + HO2                                         
        chem["DJ"].data.loc[:, :, :, ts, 34] = J.loc[:, :, :, ts, 4]*0.02*0.36   
         
        # Photol Reaction (35) UDCARB11 = RN10O2 + HO2                                            
        chem["DJ"].data.loc[:, :, :, ts, 35] = J.loc[:, :, :, ts, 4]*0.02*0.55   
         
        # Photol Reaction (36) UDCARB11 = ANHY + HO2 + CH3O2                                      
        chem["DJ"].data.loc[:, :, :, ts, 36] = J.loc[:, :, :, ts, 4]*0.02*0.45   
         
        # Photol Reaction (37) UDCARB14 = RN13O2 + HO2                                            
        chem["DJ"].data.loc[:, :, :, ts, 37] = J.loc[:, :, :, ts, 4]*0.02*0.55   
         
        # Photol Reaction (38) UDCARB14 = ANHY + HO2 + C2H5O2                                     
        chem["DJ"].data.loc[:, :, :, ts, 38] = J.loc[:, :, :, ts, 4]*0.02*0.45   
         
        # Photol Reaction (39) TNCARB26 = RTN26O2 + HO2                                           
        chem["DJ"].data.loc[:, :, :, ts, 39] = J.loc[:, :, :, ts, 15]             
         
        # Photol Reaction (40) TNCARB10 = CH3CO3 + CH3CO3 + CO                                    
        chem["DJ"].data.loc[:, :, :, ts, 40] = J.loc[:, :, :, ts, 35]*0.5        
         
        # Photol Reaction (41) CH3NO3 = HCHO + HO2 + NO2                                          
        chem["DJ"].data.loc[:, :, :, ts, 41] = J.loc[:, :, :, ts, 51]                        
         
        # Photol Reaction (42) C2H5NO3 = CH3CHO + HO2 + NO2                                       
        chem["DJ"].data.loc[:, :, :, ts, 42] = J.loc[:, :, :, ts, 52]                        
         
        # Photol Reaction (43) RN10NO3 = C2H5CHO + HO2 + NO2                                      
        chem["DJ"].data.loc[:, :, :, ts, 43] = J.loc[:, :, :, ts, 53]                        
         
        # Photol Reaction (44) IC3H7NO3 = CH3COCH3 + HO2 + NO2                                    
        chem["DJ"].data.loc[:, :, :, ts, 44] = J.loc[:, :, :, ts, 54]                        
         
        # Photol Reaction (45) RN13NO3 =  CH3CHO + C2H5O2 + NO2                                   
        chem["DJ"].data.loc[:, :, :, ts, 45] = J.loc[:, :, :, ts, 53]*BR01               
         
        # Photol Reaction (46) RN13NO3 =  CARB11A + HO2 + NO2                                     
        chem["DJ"].data.loc[:, :, :, ts, 46] = J.loc[:, :, :, ts, 53]*(1-BR01)           
         
        # Photol Reaction (47) RN16NO3 = RN15O2 + NO2                                             
        chem["DJ"].data.loc[:, :, :, ts, 47] = J.loc[:, :, :, ts, 53]                        
         
        # Photol Reaction (48) RN19NO3 = RN18O2 + NO2                                             
        chem["DJ"].data.loc[:, :, :, ts, 48] = J.loc[:, :, :, ts, 53]                        
         
        # Photol Reaction (49) RA13NO3 = CARB3 + UDCARB8 + HO2 + NO2                              
        chem["DJ"].data.loc[:, :, :, ts, 49] = J.loc[:, :, :, ts, 54]                    
         
        # Photol Reaction (50) RA16NO3 = CARB3 + UDCARB11 + HO2 + NO2                             
        chem["DJ"].data.loc[:, :, :, ts, 50] = J.loc[:, :, :, ts, 54]                    
         
        # Photol Reaction (51) RA19NO3 = CARB6 + UDCARB11 + HO2 + NO2                             
        chem["DJ"].data.loc[:, :, :, ts, 51] = J.loc[:, :, :, ts, 54]                    
         
        # Photol Reaction (52) RTX24NO3 = TXCARB22 + HO2 + NO2                                    
        chem["DJ"].data.loc[:, :, :, ts, 52] = J.loc[:, :, :, ts, 54]                    
         
        # Photol Reaction (53) CH3OOH = HCHO + HO2 + OH                                           
        chem["DJ"].data.loc[:, :, :, ts, 53] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (54) C2H5OOH = CH3CHO + HO2 + OH                                        
        chem["DJ"].data.loc[:, :, :, ts, 54] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (55) RN10OOH = C2H5CHO + HO2 + OH                                       
        chem["DJ"].data.loc[:, :, :, ts, 55] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (56) IC3H7OOH = CH3COCH3 + HO2 + OH                                     
        chem["DJ"].data.loc[:, :, :, ts, 56] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (57) RN13OOH =  CH3CHO + C2H5O2 + OH                                    
        chem["DJ"].data.loc[:, :, :, ts, 57] = J.loc[:, :, :, ts, 41]*BR01         
         
        # Photol Reaction (58) RN13OOH =  CARB11A + HO2 + OH                                      
        chem["DJ"].data.loc[:, :, :, ts, 58] = J.loc[:, :, :, ts, 41]*(1-BR01)     
         
        # Photol Reaction (59) RN16OOH = RN15AO2 + OH                                             
        chem["DJ"].data.loc[:, :, :, ts, 59] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (60) RN19OOH = RN18AO2 + OH                                             
        chem["DJ"].data.loc[:, :, :, ts, 60] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (61) CH3CO3H = CH3O2 + OH                                               
        chem["DJ"].data.loc[:, :, :, ts, 61] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (62) C2H5CO3H = C2H5O2 + OH                                             
        chem["DJ"].data.loc[:, :, :, ts, 62] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (63) HOCH2CO3H = HCHO + HO2 + OH                                        
        chem["DJ"].data.loc[:, :, :, ts, 63] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (64) RN8OOH = C2H5O2 + OH                                               
        chem["DJ"].data.loc[:, :, :, ts, 64] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (65) RN11OOH = RN10O2 + OH                                              
        chem["DJ"].data.loc[:, :, :, ts, 65] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (66) RN14OOH = RN13O2 + OH                                              
        chem["DJ"].data.loc[:, :, :, ts, 66] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (67) RN17OOH = RN16O2 + OH                                              
        chem["DJ"].data.loc[:, :, :, ts, 67] = J.loc[:, :, :, ts, 41]                        
         
        # Photol Reaction (68) RU14OOH = UCARB12 + HO2 + OH                                       
        chem["DJ"].data.loc[:, :, :, ts, 68] = J.loc[:, :, :, ts, 41]*0.252              
         
        # Photol Reaction (69) RU14OOH = UCARB10 + HCHO + HO2 + OH                                
        chem["DJ"].data.loc[:, :, :, ts, 69] = J.loc[:, :, :, ts, 41]*0.748              
         
        # Photol Reaction (70) RU12OOH = CARB6 + HOCH2CHO + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 70] = J.loc[:, :, :, ts, 41]                   
         
        # Photol Reaction (71) RU10OOH = CH3CO3 + HOCH2CHO + OH                                   
        chem["DJ"].data.loc[:, :, :, ts, 71] = J.loc[:, :, :, ts, 41]                   
         
        # Photol Reaction (72) NRU14OOH = NUCARB12 + HO2 + OH                                     
        chem["DJ"].data.loc[:, :, :, ts, 72] = J.loc[:, :, :, ts, 41]                   
         
        # Photol Reaction (73) NRU12OOH = NOA + CO + HO2 + OH                                     
        chem["DJ"].data.loc[:, :, :, ts, 73] = J.loc[:, :, :, ts, 41]                   
         
        # Photol Reaction (74) HOC2H4OOH = HCHO + HCHO + HO2 + OH                                 
        chem["DJ"].data.loc[:, :, :, ts, 74] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (75) RN9OOH = CH3CHO + HCHO + HO2 + OH                                  
        chem["DJ"].data.loc[:, :, :, ts, 75] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (76) RN12OOH = CH3CHO + CH3CHO + HO2 + OH                               
        chem["DJ"].data.loc[:, :, :, ts, 76] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (77) RN15OOH = C2H5CHO + CH3CHO + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 77] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (78) RN18OOH = C2H5CHO + C2H5CHO + HO2 + OH                             
        chem["DJ"].data.loc[:, :, :, ts, 78] = J.loc[:, :, :, ts, 41]                 
         
        # Photol Reaction (79) NRN6OOH = HCHO + HCHO + NO2 + OH                                   
        chem["DJ"].data.loc[:, :, :, ts, 79] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (80) NRN9OOH = CH3CHO + HCHO + NO2 + OH                                 
        chem["DJ"].data.loc[:, :, :, ts, 80] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (81) NRN12OOH = CH3CHO + CH3CHO + NO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 81] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (82) RA13OOH = CARB3 + UDCARB8 + HO2 + OH                               
        chem["DJ"].data.loc[:, :, :, ts, 82] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (83) RA16OOH = CARB3 + UDCARB11 + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 83] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (84) RA19OOH = CARB6 + UDCARB11 + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 84] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (85) RTN28OOH = TNCARB26 + HO2 + OH                                     
        chem["DJ"].data.loc[:, :, :, ts, 85] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (86) NRTN28OOH = TNCARB26 + NO2 + OH                                    
        chem["DJ"].data.loc[:, :, :, ts, 86] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (87) RTN26OOH = RTN25O2 + OH                                            
        chem["DJ"].data.loc[:, :, :, ts, 87] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (88) RTN25OOH = RTN24O2 + OH                                            
        chem["DJ"].data.loc[:, :, :, ts, 88] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (89) RTN24OOH = RTN23O2 + OH                                            
        chem["DJ"].data.loc[:, :, :, ts, 89] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (90) RTN23OOH = CH3COCH3 + RTN14O2 + OH                                 
        chem["DJ"].data.loc[:, :, :, ts, 90] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (91) RTN14OOH = TNCARB10 + HCHO + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 91] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (92) RTN10OOH = RN8O2 + CO + OH                                         
        chem["DJ"].data.loc[:, :, :, ts, 92] = J.loc[:, :, :, ts, 41]             
         
        # Photol Reaction (93) RTX28OOH = TXCARB24 + HCHO + HO2 + OH                              
        chem["DJ"].data.loc[:, :, :, ts, 93] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (94) RTX24OOH = TXCARB22 + HO2 + OH                                     
        chem["DJ"].data.loc[:, :, :, ts, 94] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (95) RTX22OOH = CH3COCH3 + RN13O2 + OH                                  
        chem["DJ"].data.loc[:, :, :, ts, 95] = J.loc[:, :, :, ts, 41]                  
         
        # Photol Reaction (96) NRTX28OOH = TXCARB24 + HCHO + NO2 + OH                             
        chem["DJ"].data.loc[:, :, :, ts, 96] = J.loc[:, :, :, ts, 41]

      
    def calc_aerosol(self, ts):
        """Calculate aerosol masses for each grid cell and timestep"""
    
        # Get array of concs for this timestep
        Y = self.chem["Y"].data.loc[:, :, :, ts, :]
        #print(self.chem["soa"].data)
        # Calculate secondary organic aerosol mass
        self.chem["soa"].data.loc[:, :, :, ts] = Y.sel(species='P2604') * 3.574E-10 + Y.sel(species='P4608') * 3.574E-10 + \
            Y.sel(species='P2631') * 3.059E-10 + Y.sel(species='P4610') * 3.093E-10 + Y.sel(species='P2605') * 3.093E-10 + \
            Y.sel(species='P2630') * 3.325E-10 + Y.sel(species='P2629') * 4.072E-10 + Y.sel(species='P2632') * 2.860E-10 + \
            Y.sel(species='P2637') * 3.391E-10 + Y.sel(species='P3612') * 2.310E-10 + Y.sel(species='P3613') * 2.543E-10 + \
            Y.sel(species='P3442') * 1.628E-10 + Y.sel(species='P2007') * 2.493E-10
        
        # Calculate organic matter mass
        self.chem["mom"].data.loc[:, :, :, ts] = Y.sel(species='EMPOA') +  self.params["bgoam"] + self.chem["soa"].data.loc[:, :, :, ts]
        

    def _f2py_import(self):
        chem = self.chem

        # Chem variables
        chem["Y"].data.values = boxm_for.boxm.y
        #boxm_for.boxm.yp = chem["YP"].data.values
        chem["RC"].data.values = boxm_for.boxm.rc
        chem["DJ"].data.values = boxm_for.boxm.dj
        chem["EM"].data.values = boxm_for.boxm.em
        chem["FL"].data.values = boxm_for.boxm.fl
        print("chem 5d done")
        chem["J"].data.values = boxm_for.boxm.j
        chem["soa"].data.values = boxm_for.boxm.soa
        chem["mom"].data.values = boxm_for.boxm.mom
        chem["BR01"].data.values = boxm_for.boxm.br01
        chem["RO2"].data.values = boxm_for.boxm.ro2

        # # Deallocate all variables once done to clear memory in Fortran
        # boxm_for.boxm.temp = None
        # boxm_for.boxm.pressure = None
        # boxm_for.boxm.spec_hum = None
        # boxm_for.boxm.m = None
        # boxm_for.boxm.h2o = None
        # boxm_for.boxm.o2 = None

        # boxm_for.boxm.y = None
        # #boxm_for.boxm.yp = None
        # boxm_for.boxm.rc = None
        # boxm_for.boxm.dj = None
        # boxm_for.boxm.em = None
        # boxm_for.boxm.fl = None

        # boxm_for.boxm.j = None
        # boxm_for.boxm.soa = None
        # boxm_for.boxm.mom = None
        # boxm_for.boxm.br01 = None
        # boxm_for.boxm.ro2 = None


    # ----------------
    # Output functions
    # ----------------
