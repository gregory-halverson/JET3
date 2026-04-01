"""
Generate uncertainty quantification (UQ) for calibrated Air Temperature (Ta_C).

This module provides a function to estimate the ±1-sigma uncertainty of calibrated
air temperature estimates using OLS regression coefficients derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

from typing import Union
import numpy as np
import pandas as pd
from pathlib import Path

from rasters import Raster


def generate_Ta_C_calibrated_UQ(
    NDVI: Union[Raster, np.ndarray, float],
    ST_C: Union[Raster, np.ndarray, float],
    SZA_deg: Union[Raster, np.ndarray, float],
    albedo: Union[Raster, np.ndarray, float],
    canopy_height_meters: Union[Raster, np.ndarray, float],
    elevation_m: Union[Raster, np.ndarray, float],
    emissivity: Union[Raster, np.ndarray, float],
    wind_speed_mps: Union[Raster, np.ndarray, float],
) -> Union[Raster, np.ndarray]:
    """
    Generate ±1-sigma uncertainty quantification for calibrated air temperature estimates.
    
    This function applies an OLS regression model trained on validation data to predict
    the expected absolute error (uncertainty) of calibrated Ta_C estimates using
    only remote sensing inputs.
    
    Parameters
    ----------
    NDVI : Union[Raster, np.ndarray, float]
        Normalized Difference Vegetation Index. Can be a Raster, n-dimensional array, or scalar.
    ST_C : Union[Raster, np.ndarray, float]
        Surface Temperature in Celsius
    SZA_deg : Union[Raster, np.ndarray, float]
        Solar Zenith Angle in degrees
    albedo : Union[Raster, np.ndarray, float]
        Surface albedo
    canopy_height_meters : Union[Raster, np.ndarray, float]
        Canopy height in meters
    elevation_m : Union[Raster, np.ndarray, float]
        Elevation in meters
    emissivity : Union[Raster, np.ndarray, float]
        Surface emissivity
    wind_speed_mps : Union[Raster, np.ndarray, float]
        Wind speed in meters per second
    
    Returns
    -------
    Union[Raster, np.ndarray]
        The ±1-sigma uncertainty magnitude for each input observation. Type matches input type.
        Uncertainty values are guaranteed to be non-negative.
        
    Examples
    --------
    >>> import numpy as np
    >>> from JET3.generate_Ta_C_calibrated_UQ import generate_Ta_C_calibrated_UQ
    >>> 
    >>> # Example with 10 samples
    >>> NDVI = np.array([0.5, 0.6, 0.7, ...])
    >>> ST_C = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Generate calibrated UQ
    >>> uq = generate_Ta_C_calibrated_UQ(NDVI, ST_C, SZA_deg, albedo,
    ...                                   canopy_height_meters, elevation_m,
    ...                                   emissivity, wind_speed_mps)
    >>> 
    >>> # Use with calibrated estimates
    >>> calibrated_ta_c = np.array([24.5, 25.2, 26.1, ...])
    >>> lower_bound = calibrated_ta_c - uq
    >>> upper_bound = calibrated_ta_c + uq
    
    Notes
    -----
    - Model Performance: R² = 0.0789, RMSE = 1.3780, MAE = 1.0622
    - Input arrays should have compatible shapes (normalized via broadcasting)
    - Input arrays may contain NaN values; output will be NaN at those positions
    - Coefficients were derived from ECOv002 cal/val dataset
    - These coefficients predict uncertainty of calibrated values (after error correction)
    - Raster outputs maintain the geometry of the input Raster
    """
    # Track original input type and geometry
    original_ndvi_is_raster = isinstance(NDVI, Raster)
    original_geometry = getattr(NDVI, 'geometry', None) if original_ndvi_is_raster else None
    
    # Convert all inputs to arrays
    ndvi_arr = np.asarray(NDVI, dtype=float)
    st_c_arr = np.asarray(ST_C, dtype=float)
    sza_deg_arr = np.asarray(SZA_deg, dtype=float)
    albedo_arr = np.asarray(albedo, dtype=float)
    canopy_height_arr = np.asarray(canopy_height_meters, dtype=float)
    elevation_arr = np.asarray(elevation_m, dtype=float)
    emissivity_arr = np.asarray(emissivity, dtype=float)
    wind_speed_arr = np.asarray(wind_speed_mps, dtype=float)
    
    # Broadcast all arrays to same shape
    (ndvi_arr, st_c_arr, sza_deg_arr, albedo_arr, 
     canopy_height_arr, elevation_arr, emissivity_arr, wind_speed_arr) = np.broadcast_arrays(
        ndvi_arr, st_c_arr, sza_deg_arr, albedo_arr,
        canopy_height_arr, elevation_arr, emissivity_arr, wind_speed_arr
    )
    
    # Record original shape and flatten all arrays for processing
    original_shape = ndvi_arr.shape
    ndvi_arr = ndvi_arr.flatten()
    st_c_arr = st_c_arr.flatten()
    sza_deg_arr = sza_deg_arr.flatten()
    albedo_arr = albedo_arr.flatten()
    canopy_height_arr = canopy_height_arr.flatten()
    elevation_arr = elevation_arr.flatten()
    emissivity_arr = emissivity_arr.flatten()
    wind_speed_arr = wind_speed_arr.flatten()
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "Ta_C_calibrated_UQ_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure Ta_C_calibrated_UQ_coefficients.csv is in the JET3 package directory."
        )
    
    coef_df = pd.read_csv(coef_path)
    
    # Extract intercept
    intercept_row = coef_df[coef_df['Variable'] == 'Intercept']
    if intercept_row.empty:
        raise ValueError("Intercept coefficient not found in coefficient file")
    intercept = intercept_row['Coefficient'].values[0]
    
    # Extract coefficients for predictor variables
    predictor_coefs = coef_df[coef_df['Variable'] != 'Intercept'].copy()
    
    # Build predictor dictionary
    predictors = {
        'NDVI': ndvi_arr,
        'ST_C': st_c_arr,
        'SZA_deg': sza_deg_arr,
        'albedo': albedo_arr,
        'canopy_height_meters': canopy_height_arr,
        'elevation_m': elevation_arr,
        'emissivity': emissivity_arr,
        'wind_speed_mps': wind_speed_arr,
    }
    
    # Check array lengths match
    n = len(ndvi_arr)
    for var_name, arr in predictors.items():
        if len(arr) != n:
            raise ValueError(
                f"Input array length mismatch: {var_name} has length {len(arr)}, "
                f"but NDVI has length {n}"
            )
    
    # Create mask for valid (non-NaN) values across all inputs
    valid_mask = np.ones(n, dtype=bool)
    for arr in predictors.values():
        valid_mask &= ~np.isnan(arr)
    
    # Initialize output with NaN
    uq = np.full(n, np.nan, dtype=float)
    
    # Only calculate for valid positions
    if valid_mask.any():
        # Apply OLS regression: UQ = intercept + sum(coef_i * predictor_i)
        uq_valid = np.full(valid_mask.sum(), intercept, dtype=float)
        
        for _, row in predictor_coefs.iterrows():
            var = row['Variable']
            coef = row['Coefficient']
            if var not in predictors:
                raise ValueError(f"Predictor '{var}' from coefficients not found in input parameters")
            uq_valid += coef * predictors[var][valid_mask]
        
        # Ensure non-negative uncertainty
        uq_valid = np.maximum(uq_valid, 0)
        
        # Assign to output
        uq[valid_mask] = uq_valid
    
    # Reshape result back to original shape
    uq = uq.reshape(original_shape)
    
    # Wrap in Raster if input was Raster
    if original_ndvi_is_raster:
        import rasters as rt
        uq = rt.Raster(uq, geometry=original_geometry)
    
    return uq
