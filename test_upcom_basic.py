"""Basic test script for UPCOM model."""

import numpy as np
import xarray as xr
from pycontrails.core.met import MetDataset
from pycontrails.models.upcom import UPCOM, UPCOMParams, sac

def create_test_met():
    """Create a simple test MetDataset."""
    rng = np.random.default_rng(0)

    # Create coordinate arrays
    time = np.array(['2022-01-01T00:00:00'], dtype='datetime64[ns]')
    level = np.array([200, 250, 300])  # hPa
    latitude = np.linspace(30, 50, 5)
    longitude = np.linspace(-100, -80, 5)
    
    # Create meshgrid
    coords = {
        'time': time,
        'level': level,
        'latitude': latitude,
        'longitude': longitude
    }
    
    shape = (len(time), len(level), len(latitude), len(longitude))
    
    # Create realistic temperature field (decreasing with altitude)
    # At cruise altitudes: ~220 K at 200 hPa, ~240 K at 300 hPa
    T_base = np.array([220, 230, 240])  # K, for each level
    air_temperature = np.zeros(shape)
    for i, T in enumerate(T_base):
        air_temperature[0, i, :, :] = T + rng.normal(size=(len(latitude), len(longitude))) * 2
    
    # Create specific humidity field
    # Typical values: ~1e-5 to 1e-4 kg/kg at cruise altitude
    specific_humidity = rng.uniform(3e-5, 8e-5, shape)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'air_temperature': (['time', 'level', 'latitude', 'longitude'], air_temperature),
            'specific_humidity': (['time', 'level', 'latitude', 'longitude'], specific_humidity),
        },
        coords=coords
    )
    
    return MetDataset(ds)

def main():
    """Run basic UPCOM test."""
    print("Creating test meteorology data...")
    met = create_test_met()
    
    print(f"Met dataset shape: {met.data['air_temperature'].shape}")
    print(f"Temperature range: {met.data['air_temperature'].values.min():.1f} - {met.data['air_temperature'].values.max():.1f} K")
    print(f"Specific humidity range: {met.data['specific_humidity'].values.min():.2e} - {met.data['specific_humidity'].values.max():.2e} kg/kg")
    
    print("\nInitializing UPCOM model...")
    model = UPCOM(met, params=UPCOMParams(rhi_threshold=1.0))
    
    print("Running UPCOM.eval()...")
    result = model.eval()
    
    print("\nResults:")
    print(f"Available fields: {list(result.data.data_vars)}")
    
    # Check RHi
    rhi = result.data['rhi'].values
    print(f"\nRHi range: {np.nanmin(rhi):.3f} - {np.nanmax(rhi):.3f}")
    print(f"Points with RHi > 1.0: {np.sum(rhi > 1.0)} / {rhi.size}")
    
    # Check ISSR
    issr = result.data['issr'].values
    print(f"\nISSR points: {np.sum(issr > 0)} / {issr.size} ({100*np.sum(issr)/issr.size:.1f}%)")
    
    # Check contrail thresholds
    T_contr = result.data['T_contr'].values
    print(f"\nT_contr range: {np.nanmin(T_contr):.1f} - {np.nanmax(T_contr):.1f} K")
    
    RH_contr = result.data['RH_contr'].values
    print(f"RH_contr range: {np.nanmin(RH_contr):.3f} - {np.nanmax(RH_contr):.3f}")

    # Check SAC and both optional helper return modes
    sac_output = result.data['sac']
    sac_points = np.sum(sac_output.values > 0)
    print(f"\nSAC points: {sac_points} / {sac_output.size} ({100*sac_points/sac_output.size:.1f}%)")

    sac_only = sac(
        result.data['air_temperature'],
        result.data['rh_liquid'],
        result.data['air_pressure'],
        model.params,
    )
    np.testing.assert_array_equal(sac_only, sac_output)

    sac_flag, sac_G, sac_T_contr, sac_RH_contr = sac(
        result.data['air_temperature'],
        result.data['rh_liquid'],
        result.data['air_pressure'],
        model.params,
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(sac_flag, sac_output)
    np.testing.assert_array_equal(sac_G, result.data['G'])
    np.testing.assert_array_equal(sac_T_contr, result.data['T_contr'])
    np.testing.assert_array_equal(sac_RH_contr, result.data['RH_contr'])
    
    # Check potential persistent contrails
    ppc = result.data['potential_persistent_contrail'].values
    print(f"\nPotential persistent contrail points: {np.sum(ppc > 0)} / {ppc.size} ({100*np.sum(ppc)/ppc.size:.1f}%)")

    legacy_ppc = (
        (result.data['rhi'] > 1.0)
        & (result.data['air_temperature'] < result.data['T_contr'])
        & (result.data['rh_liquid'] > result.data['RH_contr'])
    ).astype(result.data['rhi'].dtype)
    np.testing.assert_array_equal(ppc, legacy_ppc)
    np.testing.assert_array_equal(
        result.data['potential_persistent_contrail'],
        ((result.data['issr'] == 1) & (result.data['sac'] == 1)).astype(
            result.data['rhi'].dtype
        ),
    )
    
    # Check G parameter
    G = result.data['G'].values
    print(f"\nG parameter range: {np.nanmin(G):.6f} - {np.nanmax(G):.6f}")
    print(f"G > 0.053 (valid for T_contr): {np.sum(G > 0.053)} / {G.size}")
    
    print("\n✓ UPCOM model test completed successfully!")
    print("\nModel attributes:")
    for key, value in result.attrs.items():
        if key.startswith('upcom_'):
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()