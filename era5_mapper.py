"""
This file contains functions to reduce the era5 dataset to regions defined by shapefiles.
"""

from energy_type import EnergyType
import config
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from enum import Enum

def create_era5_region():
    """
    The function creates an era5 data set that has been reduced to the regions of the shapefiles.
    The path to the resource files is defined in the config file.
    Before the function can be executed, the complete era5 dataset must be downloaded.
    """

    era5_eu_2013 = Path(config.paths["era5_eu_2013"])
    if era5_eu_2013.is_file():
        print(f'The file {config.paths["era5_eu_2013"]} exists')

        # Load the input data
        era_data = xr.open_dataset(filename_or_obj=config.paths["era5_eu_2013"], engine="netcdf4")
        gdf_onshore = gpd.read_file(config.paths["onshore_shape"])
        gdf_offshore = gpd.read_file(config.paths["offshore_shape"])

        regions_onshore, regions_offshore = _map_coordinates_to_regions(era_data, gdf_onshore, gdf_offshore)
        _create_era5_region_data(era_data, gdf_onshore, gdf_offshore, regions_onshore, regions_offshore)

    else:
        print(f'The file {config.paths["era5_eu_2013"]} does not exist.')
        print("The file must first be downloaded from the website: https://zenodo.org/record/4709858#.YZUVdCYo8WM")


def _map_coordinates_to_regions(era_data, gdf_onshore, gdf_offshore):
    """
    Helper function that maps the coordinates from the era_data to the regions described by the shapefiles.

    :param era_data: Full era5 dataset.
    :param gdf_onshore: Onshore Shapefile.
    :param gdf_offshore: Offshore Shapefile.
    :return: A list of all the regions and the coordinates that lies within this region.
    """
    dim_x, dim_y, dim_t = era_data.sizes.values()

    # list of all coordinates that are within the regions given by the shapefiles
    regions_onshore = [[] for _ in range(gdf_onshore.shape[0])]
    regions_offshore = [[] for _ in range(gdf_offshore.shape[0])]

    print("Mapping coordinates to their regions given by the shapefiles ...")
    i = 0

    for y in range(dim_y):
        for x in range(dim_x):
            point = Point(era_data.coords['x'].values[x], era_data.coords['y'].values[y])

            i += 1
            if i % 1000 == 0:
                print("Checking " + str(point) + " " + str(i) + " of " + str(dim_y * dim_x))

            for region_idx in range(gdf_onshore.shape[0]):
                polygon = gdf_onshore.iloc[region_idx].geometry
                if point.within(polygon):
                    regions_onshore[region_idx].append((point.x, point.y))
                    break

            for region_idx in range(gdf_offshore.shape[0]):
                polygon = gdf_offshore.iloc[region_idx].geometry
                if point.within(polygon):
                    regions_offshore[region_idx].append((point.x, point.y))
                    break

    return regions_onshore, regions_offshore


def _create_era5_region_data(era_data, gdf_onshore, gdf_offshore, regions_onshore, regions_offshore):
    """
    Helper function that takes the average of all coordinates within a region and creates a new xarray dataset.
    :param era_data: Full era5 dataset.
    :param gdf_onshore: Onshore Shapefile.
    :param gdf_offshore: Offshore Shapefile.
    :param regions_onshore: A list of all the regions and the coordinates that lies within this region.
    :param regions_offshore: A list of all the regions and the coordinates that lies within this region.
    """
    print("Creating era5 data for the regions. This process can take a very long time.")

    region_coords = regions_onshore + regions_offshore

    # Coordinates. Adding "on" and "off" to the name to avoid duplicates in the offshore and onshore region names.
    times = era_data["time"].values
    regions_on = gdf_onshore["name"].values
    for i in range(regions_on.shape[0]):
        regions_on[i] = regions_on[i] + " on"
    regions_off = gdf_offshore["name"].values
    for i in range(regions_off.shape[0]):
        regions_off[i] = regions_off[i] + " off"
    regions = np.concatenate((regions_on, regions_off))

    # Dimension/Coordinates sizes
    n_time = times.shape[0]
    n_regions = regions.shape[0]

    # Initial data variables with 0
    height = np.zeros((n_regions, n_time))
    wnd100m = np.zeros((n_regions, n_time))
    roughness = np.zeros((n_regions, n_time))
    influx_toa = np.zeros((n_regions, n_time))
    influx_direct = np.zeros((n_regions, n_time))
    influx_diffuse = np.zeros((n_regions, n_time))
    albedo = np.zeros((n_regions, n_time))
    temperature = np.zeros((n_regions, n_time))
    soil_temperature = np.zeros((n_regions, n_time))
    runoff = np.zeros((n_regions, n_time))

    for region_idx in range(n_regions):

        print("Creating Dataset for region " + str(region_idx + 1) + " (" + str(
            len(region_coords[region_idx])) + " points) of " + str(n_regions))

        # height calculation
        height_sum = 0
        for point in region_coords[region_idx]:
            val = era_data["height"].sel(x=point[0], y=point[1]).item(0)
            height_sum += val
        height_mean = height_sum / len(region_coords[region_idx])
        height[region_idx, :] = height_mean

        for point in region_coords[region_idx]:
            val_wnd = era_data["wnd100m"].sel(x=point[0], y=point[1]).values
            val_roughness = era_data["roughness"].sel(x=point[0], y=point[1]).values
            val_influx_toa = era_data["influx_toa"].sel(x=point[0], y=point[1]).values
            val_influx_direct = era_data["influx_direct"].sel(x=point[0], y=point[1]).values
            val_influx_diffuse = era_data["influx_diffuse"].sel(x=point[0], y=point[1]).values
            val_albedo = era_data["albedo"].sel(x=point[0], y=point[1]).values
            val_temperature = era_data["temperature"].sel(x=point[0], y=point[1]).values
            val_soil_temperature = era_data["soil temperature"].sel(x=point[0], y=point[1]).values
            val_runoff = era_data["runoff"].sel(x=point[0], y=point[1]).values

            wnd100m[region_idx, :] += val_wnd
            roughness[region_idx, :] += val_roughness
            influx_toa[region_idx, :] += val_influx_toa
            influx_direct[region_idx, :] += val_influx_direct
            influx_diffuse[region_idx, :] += val_influx_diffuse
            albedo[region_idx, :] += val_albedo
            temperature[region_idx, :] += val_temperature
            soil_temperature[region_idx, :] += val_soil_temperature
            runoff[region_idx, :] += val_runoff

        wnd100m[region_idx, :] /= len(region_coords[region_idx])
        roughness[region_idx, :] /= len(region_coords[region_idx])
        influx_toa[region_idx, :] /= len(region_coords[region_idx])
        influx_direct[region_idx, :] /= len(region_coords[region_idx])
        influx_diffuse[region_idx, :] /= len(region_coords[region_idx])
        albedo[region_idx, :] /= len(region_coords[region_idx])
        temperature[region_idx, :] /= len(region_coords[region_idx])
        soil_temperature[region_idx, :] /= len(region_coords[region_idx])
        runoff[region_idx, :] /= len(region_coords[region_idx])

    ds = xr.Dataset(
        data_vars=dict(
            height=(["region", "time"], height),
            wnd100m=(["region", "time"], wnd100m),
            roughness=(["region", "time"], roughness),
            influx_toa=(["region", "time"], influx_toa),
            influx_direct=(["region", "time"], influx_direct),
            influx_diffuse=(["region", "time"], influx_diffuse),
            albedo=(["region", "time"], albedo),
            temperature=(["region", "time"], temperature),
            soil_temperature=(["region", "time"], soil_temperature),
            runoff=(["region", "time"], runoff),
        ),
        coords=dict(
            region=(["region"], regions),
            time=(["time"], times),
        ),
        attrs=dict(
            description="Era5 data with mean value of the coordinates within a region",
        ),
    )

    ds.to_netcdf(config.paths["era5_regions"])


def get_era5_region_name(region_name: str, energy_type: EnergyType) -> str:
    """
    Returns the name or string that addresses the given region and energy type which can be used to address the data in the feature data set (era5)
    :param region_name: name of the region
    :param energy_type: the uses energy type in that region
    :return: string that can be used to fetch data from the feature data set
    """
    era5_region_name = region_name + " 0"
    if energy_type == EnergyType.ONWIND or energy_type == EnergyType.SOLAR or energy_type == EnergyType.ROR:
        era5_region_name += " on"
    elif energy_type == EnergyType.OFFWIND_AC or energy_type == EnergyType.OFFWIND_DC:
        era5_region_name += " off"
    else:
        era5_region_name += ""
    return era5_region_name