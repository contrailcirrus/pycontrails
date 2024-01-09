import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from pycontrails.datalib.ecmwf import ERA5
from init_chem import CHEM
from chem import ChemDataset
from boxm import BoxModel
from pycontrails.core import datalib
from pycontrails.physics import geo, thermo, units, constants
import numpy as np
import itertools
import os
import tempfile


# This file needs to generate a common input to feed to both original BOXM and new f2py implementation. The aim is to generate two sets  of outputs (species concs over time), and to diff these outputs to see how different they are to each other.

# Hard coded inputs
time = ("2022-12-20 12:00:00", "2022-12-21 18:00:00")
lon_bounds = (-135, -134) 
lat_bounds = (-45, -44)
alt_bounds = (12000, 12500)
horiz_res = 1
vert_res = 500
ts_met = "6H"
ts_disp = "1min"
ts_chem = "20s"

species = np.loadtxt('species_num.txt', dtype=str)

def main():
        
        # Create met, bg_chem, emi datasets and send to files
        lats, lons, alts, timesteps, met, bg_chem, emi = preprocess()

        run_boxm(met, bg_chem, timesteps, lats, lons, alts)




        # Run original boxm
        #run_boxm(met, chem, timesteps, lats, lons, alts)

        # lats, lons, alts, timesteps, met, chem = preprocess()
        
        # # Run f2py implementation
        # run_f2py(met, chem, emi)
        
        
        # lats, lons, alts, timesteps, met, chem = preprocess()
        
        # Run f2py compare implementation
        run_f2py_compare(met, bg_chem, emi)


def preprocess():
        met_pressure_levels = np.array([400, 300, 200, 100])

        lons = np.arange(lon_bounds[0], lon_bounds[1], horiz_res)
        lats = np.arange(lat_bounds[0], lat_bounds[1], horiz_res)
        alts = np.arange(alt_bounds[0], alt_bounds[1], vert_res)
        timesteps = datalib.parse_timesteps(time, freq=ts_chem)

        # Import met data from ERA5
        era5 = ERA5(
                time=time,
                timestep_freq=ts_met,
                variables=[
                        "t",
                        "q",
                        "relative_humidity"
                ],
                grid=1.0,
                url="https://cds.climate.copernicus.eu/api/v2",
                key="171715:93148f05-a469-43a8-ae25-44c8eba81e90",
                pressure_levels=met_pressure_levels
        )

        # download data from ERA5 (or open from cache)
        met = era5.open_metdataset()
        met.data = met.data.transpose("latitude", "longitude", "level", "time", ...)

        met = calc_M_H2O(met)

        met = zenith(met)

        month = timesteps[0].month
        bg_chem = xr.open_dataset("species.nc").sel(month=month-1) * 1E+09
        
        emi = emissions(lats, lons, alts, timesteps)

        # Downselect and interpolate both met and chem datasets to high-res grid
        met.data = met.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), time=timesteps, method="nearest")

        bg_chem = bg_chem.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), method="nearest")
        #print(bg_chem)

        emi.data = emi.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), time=timesteps, method="nearest")

        # Convert to flattened dataframes       
        pd.set_option('display.max_rows', 500)
        met_df = met.data.to_dataframe(dim_order=['time', 'level', 'longitude', 'latitude']).reset_index()
        bg_chem_df = bg_chem.to_dataframe(dim_order=['level', 'longitude', 'latitude']).reset_index()
        emi_df = emi.data.to_dataframe(dim_order=['time', 'level', 'longitude', 'latitude']).fillna(0).reset_index()
        print(emi_df)

        # Remove temporary files if they exist
        if os.path.exists("met_df.csv"):
                os.remove("met_df.csv")
        if os.path.exists("bg_chem_df.csv"):
                os.remove("bg_chem_df.csv")
        if os.path.exists("emi_df.csv"):
                os.remove("emi_df.csv")

        # Write DataFrame 1 to the temporary file
        met_df.to_csv("met_df.csv", index=False)

        # Write DataFrame 2 to the temporary file
        bg_chem_df.to_csv("bg_chem_df.csv", index=False)

        # Write DataFrame 3 to the temporary file
        emi_df.to_csv("emi_df.csv", index=False)

        return lats, lons, alts, timesteps, met, bg_chem, emi
        # chem.zenith()
        # chem.get_photol_params()
        # chem.calc_M_H2O(met)

def emissions(lats, lons, alts, timesteps):
        
        dataarrays = {}
        for s in species:
                dataarrays[s] = xr.DataArray(
                np.zeros((len(lats), len(lons), len(alts), len(timesteps))),
                dims=["latitude", "longitude", "level", "time"],
                coords={
                "latitude": lats,
                "longitude": lons,
                "level": alts,
                "time": timesteps,
                },
                name=s,
        )
                
        emi = xr.Dataset(dataarrays)
        emi = ChemDataset(emi)
        print(emi.data)
        emi.data = emi.data.transpose("latitude", "longitude", "level", "time", ...)

        return emi

def calc_M_H2O(met):

        """Calculate number density of air molecules at each pressure level M"""
        N_A = 6.022e23 # Avogadro's number
        
        # Get air density from pycontrails physics.thermo script
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)

        # Calculate number density of air (M) to feed into box model calcs
        met.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        met.data["M"] = met.data["M"].transpose("latitude", "longitude", "level", "time")
                
        # Use expand_dims to add the new "species" dimension
        #self.data["M"] = self.data["M"].expand_dims(dim={'species': self.data["species"]}, axis=4)


        # Calculate H2O number concentration to feed into box model calcs
        met.data["H2O"] = (met["specific_humidity"].data / constants.M_v) * N_A * rho_d * 1e-6 
        # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        met.data["O2"] = 2.079E-01 * met.data["M"]
        met.data["N2"] = 7.809E-01 * met.data["M"] 

        return met


def zenith(met):
        """Calculate zenith angle for each grid cell and timestep"""

        chem = self.data

        # For each timestep, calculate zenith for each grid cell in a vectorised way
        _, lon_mesh = xr.broadcast(chem.local_time, chem.longitude)
        _, lat_mesh = xr.broadcast(chem.local_time, chem.latitude)
        _, time_mesh = xr.broadcast(chem.local_time, chem.time)

        # Calculate the offset from UTC based on the longitude
        offsets = (lon_mesh / 15)  # Each hour corresponds to 15 degrees of longitude
        offsets = offsets * 3600  # Convert to seconds
        delta = offsets.astype('timedelta64[s]')

        chem["local_time"] = time_mesh + delta
        #local_time, lat_mesh = xr.broadcast(local_time, chem.latitude)
        

        # Calculate number of leap days since or before the year 2000
        if chem["local_time.year"].values.all() >= 2001:
                DL = (chem["local_time.year"].values - 2001) / 4
        else:
                DL = (chem["local_time.year"].values - 2000) / 4 - 1
        
        # Calculate Julian day of year (number of days since 1st Jan)
        DJ = chem["local_time.dayofyear"].values

        # # Calculate number of days since 1st Jan 2000
        NJD = 364.5 + (chem["local_time.year"].values - 2001) * 365.25 + DL + DJ     

        # Calculate obliquity of the ecliptic
        eps_ob = np.radians(23.439 - 0.0000004 * NJD)
        
        # Calculate mean longitude of the Sun  
        LM = np.radians(280.460 + 0.9856474 * NJD)
        
        # Calculate mean anomaly of the Sun
        GM = np.radians(357.528 + 0.9856003 * NJD)
        
        # Calculate ecliptic longitude of the Sun
        LE = LM + np.radians(1.915 * np.sin(GM) + 0.020 * np.sin(2 * GM))
        
        # Calculate solar declination angle
        d = np.arcsin(np.sin(eps_ob) * np.sin(LE))
        
        # Calculate local hour angle        
        secday = chem["local_time.hour"].values * 3600 \
        + chem["local_time.minute"].values * 60 \
        + chem["local_time.second"].values
        
        lha = (2*np.pi * (secday/86400 - 0.5))

        self.data["sza"] = np.arccos(np.cos(np.radians(lat_mesh))*np.cos(d)*np.cos(lha) + np.sin(np.radians(lat_mesh))*np.sin(d))


def run_boxm(met, bg_chem, timesteps, lats, lons, alts):
        start_time = timesteps[0]
        end_time = timesteps[-1]
        
        SPECIES = ["NO2", "NO", "O3", "CO", "CH4", "HCHO", "CH3CHO", "CH3COCH3",
                           "C2H6", "C2H4", "C3H8", "C3H6", "C2H2", "NC4H10", "TBUT2ENE",
                           "BENZENE", "TOLUENE", "OXYL", "C5H8", "H2O2", "HNO3", "C2H5CHO",
                           "CH3OH", "MEK", "CH3OOH", "PAN", "MPAN"]
        
        for lat, lon, alt in itertools.product(lats, lons, alts):
                DAY = start_time.day
                MONTH = start_time.month
                YEAR = start_time.year
                LEVEL = get_pressure_level(alt, met)
                longbox = longitude_to_longbox(lon)
                latbox = latitude_to_latbox(lat)
                RUNTIME = end_time.day - start_time.day

                M = met["M"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt), time=start_time).values.item()
                P = units.m_to_pl(alt).item()
                H2O = met["H2O"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt), time=start_time).values.item()
                TEMP = met["air_temperature"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt), time=start_time).values.item()

                BOXMinput = open("BOXMinput_" + repr(lon) + "_" + repr(lat) + "_" + repr(alt) + ".in", "w")

                BOXMinput.write(repr(DAY) + "\n" + repr(MONTH) + "\n" + repr(YEAR) + "\n" + repr(LEVEL) + "\n" + repr(longbox) + "\n" + repr(latbox) + "\n" + repr(RUNTIME) + "\n" + repr(M) + "\n" + repr(P) + "\n" + repr(H2O) + "\n" + repr(TEMP) + "\n")

                for s in SPECIES:
                        BOXMinput.write(repr(bg_chem[s].loc[lat, lon, units.m_to_pl(alt)].values.item()) + "\n")
                        #      .sel(species=s, longitude=lon, latitude=lat, time=start_time).sel(level=units.m_to_pl(alt), method="nearest").values.item()) + "\n")
                        
                        print(repr(s) + " " + repr(bg_chem[s].loc[lat, lon, units.m_to_pl(alt)].values.item()))

                        # print(repr(s) + " " + repr(chem["Y"].data.sel(species=s, longitude=lon, latitude=lat, time=start_time).sel(level=units.m_to_pl(alt), method="nearest").values.item()))
                
                BOXMinput.close()

def run_f2py(met, chem, emi):
        boxm = BoxModel(met, chem)

        boxm.eval(source=emi)

        boxm.met.data.to_netcdf("f2py_met.nc", mode="w")
        boxm.chem.data.to_netcdf("f2py_chem.nc", mode="w")

def run_f2py_compare(met, chem, emi):
        boxm = BoxModel(met, chem)

        boxm.met["air_temperature"].data.values = xr.full_like(boxm.met["air_temperature"].data, 
                                                        fill_value=met["air_temperature"].data.values[0,0,0,0])
        boxm.met["air_pressure"].data.values = xr.full_like(boxm.met["air_pressure"].data, 
                                                     fill_value=met["air_pressure"].data.values[0])

        boxm.chem["M"].data.values = xr.full_like(boxm.chem["M"].data, 
                                           fill_value=chem["M"].data.values[0,0,0,0])
        boxm.chem["H2O"].data.values = xr.full_like(boxm.chem["H2O"].data, 
                                                    fill_value=chem["H2O"].data.values[0,0,0,0])
        boxm.chem["O2"].data.values = xr.full_like(boxm.chem["O2"].data, 
                                                   fill_value=chem["O2"].data.values[0,0,0,0])
        boxm.chem["N2"].data.values = xr.full_like(boxm.chem["N2"].data, 
                                                   fill_value=chem["N2"].data.values[0,0,0,0])
                
        boxm.eval(source=emi)

        os.remove("f2py_c_met.nc")
        os.remove("f2py_c_chem.nc")
        boxm.met.data.to_netcdf("f2py_c_met.nc", mode="w")
        boxm.chem.data.to_netcdf("f2py_c_chem.nc", mode="w")


def get_pressure_level(alt, met):
        # Convert alt to pressure level (hPa)
        chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])

        # Convert altitude to pressure using a standard atmosphere model
        pressure = units.m_to_pl(alt)

        # Find the index of the closest value in the array
        idx = (np.abs(chem_pressure_levels - pressure)).argmin()

        return idx

def latitude_to_latbox(latitude):
        # Map the latitude to the range 0-1
        normalized_latitude = (latitude + 90) / 180

        # Map the normalized latitude to the range 1-72
        latbox = normalized_latitude * 36

        # Round to the nearest integer and return
        return round(latbox)

def longitude_to_longbox(longitude):
        # Map the longitude to the range 0-1
        normalized_longitude = (longitude + 180) / 360

        # Map the normalized longitude to the range 1-144
        longbox = normalized_longitude * 72

        # Round to the nearest integer and return
        return round(longbox)

if __name__=="__main__":
        main()
