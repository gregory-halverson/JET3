from datetime import date, datetime
import logging

import numpy as np
import pandas as pd
import rasters as rt
import sklearn
from check_distribution import check_distribution
from GEOS5FP import GEOS5FP
from rasters import Raster, RasterGeometry

from .exceptions import BlankOutputError

logger = logging.getLogger(__name__)


def sharpen_soil_moisture_data(
    ST_C: Raster,
    NDVI: Raster,
    albedo: Raster,
    water_mask: Raster,
    geometry: RasterGeometry,
    coarse_geometry: RasterGeometry,
    time_UTC: datetime,
    date_UTC: date,
    tile: str,
    orbit: str,
    scene: str,
    upsampling: str,
    downsampling: str,
    GEOS5FP_connection: GEOS5FP,
) -> Raster:
    """
    Sharpens soil moisture using a linear regression model.

    Args:
        ST_C: Surface temperature in Celsius (fine resolution).
        NDVI: Normalized Difference Vegetation Index (fine resolution).
        albedo: Albedo (fine resolution).
        water_mask: Water mask (fine resolution).
        geometry: Fine-resolution raster geometry.
        coarse_geometry: Coarse-resolution raster geometry.
        time_UTC: UTC timestamp of the overpass.
        date_UTC: UTC date of the overpass.
        tile: Granule tile ID.
        orbit: Orbit ID.
        scene: Scene ID.
        upsampling: Upsampling method for spatial resampling.
        downsampling: Downsampling method for spatial resampling.
        GEOS5FP_connection: An instance of the GEOS5FP connection.

    Returns:
        SM: Sharpened soil moisture.

    Raises:
        BlankOutputError: If the sharpened soil moisture output is blank.
    """
    ST_C_coarse = ST_C.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(ST_C_coarse, "ST_C_coarse", date_UTC, tile)
    NDVI_coarse = NDVI.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(NDVI_coarse, "NDVI_coarse", date_UTC, tile)
    albedo_coarse = albedo.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(albedo_coarse, "albedo_coarse", date_UTC, tile)

    SM_coarse = GEOS5FP_connection.SM(
        time_UTC=time_UTC, geometry=coarse_geometry, resampling=downsampling
    )
    check_distribution(SM_coarse, "SM_coarse", date_UTC, tile)

    coarse_samples = pd.DataFrame(
        {
            "SM": np.array(SM_coarse).ravel(),
            "ST_C": np.array(ST_C_coarse).ravel(),
            "NDVI": np.array(NDVI_coarse).ravel(),
            "albedo": np.array(albedo_coarse).ravel(),
        }
    ).dropna()

    if len(coarse_samples) == 0:
        raise ValueError(
            f"unable to retrieve coarse samples for downscaling orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC"
        )

    SM_model = sklearn.linear_model.LinearRegression()
    SM_model.fit(coarse_samples[["ST_C", "NDVI", "albedo"]], coarse_samples["SM"])
    SM_intercept = SM_model.intercept_
    ST_C_SM_coef, NDVI_SM_coef, albedo_SM_coef = SM_model.coef_
    logger.info(
        f"soil moisture regression: SM = {SM_intercept:0.2f} + {ST_C_SM_coef:0.2f} * ST_C + {NDVI_SM_coef:0.2f} * NDVI + {albedo_SM_coef:0.2f} * albedo"
    )

    SM_prediction = rt.clip(
        ST_C * ST_C_SM_coef + NDVI * NDVI_SM_coef + albedo * albedo_SM_coef + SM_intercept,
        0,
        1,
    )
    check_distribution(SM_prediction, "SM_prediction", date_UTC, tile)

    logger.info(
        f"up-sampling predicted soil moisture from {int(SM_prediction.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method"
    )
    SM_prediction_coarse = SM_prediction.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(SM_prediction_coarse, "SM_prediction_coarse", date_UTC, tile)
    SM_bias_coarse = SM_prediction_coarse - SM_coarse
    check_distribution(SM_bias_coarse, "SM_bias_coarse", date_UTC, tile)

    logger.info(
        f"down-sampling soil moisture bias from {int(SM_bias_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method"
    )
    SM_bias_smooth = SM_bias_coarse.to_geometry(geometry, resampling=downsampling)
    check_distribution(SM_bias_smooth, "SM_bias_smooth", date_UTC, tile)
    logger.info("bias-correcting soil moisture")
    SM = rt.clip(SM_prediction - SM_bias_smooth, 0, 1)
    check_distribution(SM, "SM", date_UTC, tile)

    SM_smooth = GEOS5FP_connection.SM(time_UTC=time_UTC, geometry=geometry, resampling=downsampling)
    check_distribution(SM_smooth, "SM_smooth", date_UTC, tile)
    logger.info("gap-filling soil moisture")
    SM = rt.clip(rt.where(np.isnan(SM), SM_smooth, SM), 0, 1)
    SM = rt.where(water_mask, np.nan, SM)
    check_distribution(SM, "SM", date_UTC, tile)

    logger.info(
        f"up-sampling final soil moisture from {int(SM.cell_size)}m to {int(coarse_geometry.cell_size)}m with {upsampling} method"
    )
    SM_final_coarse = SM.to_geometry(coarse_geometry, resampling=upsampling)
    check_distribution(SM_final_coarse, "SM_final_coarse", date_UTC, tile)
    SM_error_coarse = SM_final_coarse - SM_coarse
    check_distribution(SM_error_coarse, "SM_error_coarse", date_UTC, tile)
    logger.info(
        f"down-sampling soil moisture error from {int(SM_error_coarse.cell_size)}m to {int(geometry.cell_size)}m with {downsampling} method"
    )
    SM_error = rt.where(water_mask, np.nan, SM_error_coarse.to_geometry(geometry, resampling=downsampling))
    check_distribution(SM_error, "SM_error", date_UTC, tile)

    if np.all(np.isnan(SM)):
        raise BlankOutputError(
            f"blank soil moisture output for orbit {orbit} scene {scene} tile {tile} at {time_UTC} UTC"
        )

    return SM
