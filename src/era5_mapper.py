from src.energy_type import EnergyType
import config

import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point


class Era5Mapper:
    """
    This class contains functions to reduce the era5 dataset to regions defined by shapefiles.
    """

    def __init__(self):
        """
        Initializes the Era5Mapper. Loads the era5 weather dataset and the shapefiles
        """

        era5_eu_2013 = Path(config.paths["era5_eu_2013"])
        if era5_eu_2013.is_file():
            print(f'The file {config.paths["era5_eu_2013"]} exists')

            # Load the input data
            self.era_data = xr.open_dataset(filename_or_obj=config.paths["era5_eu_2013"], engine="netcdf4")
            self.gdf_onshore = gpd.read_file(config.paths["onshore_shape"])
            self.gdf_offshore = gpd.read_file(config.paths["offshore_shape"])

        else:
            print(f'The file {config.paths["era5_eu_2013"]} does not exist.')
            print("The file must first be downloaded from the website: https://zenodo.org/record/4709858#.YZUVdCYo8WM")

    def create_era5_region(self):
        """
        The function creates an era5 data set that has been reduced to the regions of the shapefiles.
        The path to the resource files is defined in the config file.
        Before the function can be executed, the complete era5 dataset must be downloaded.
        """

        regions_onshore, regions_offshore = self._map_coordinates_to_regions()
        self._create_era5_region_data(regions_onshore, regions_offshore)

    def _map_coordinates_to_regions(self):
        """
        Helper function that maps the coordinates from the era_data to the regions described by the shapefiles.

        :return: A list of all the regions and the coordinates that lies within this region.
        """

        dim_x, dim_y, dim_t = self.era_data.sizes.values()

        # list of all coordinates that are within the regions given by the shapefiles
        regions_onshore = [[] for _ in range(self.gdf_onshore.shape[0])]
        regions_offshore = [[] for _ in range(self.gdf_offshore.shape[0])]

        print("Mapping coordinates to their regions given by the shapefiles ...")
        i = 0

        for y in range(dim_y):
            for x in range(dim_x):
                point = Point(self.era_data.coords['x'].values[x], self.era_data.coords['y'].values[y])

                i += 1
                if i % 1000 == 0:
                    print("Checking " + str(point) + " " + str(i) + " of " + str(dim_y * dim_x))

                for region_idx in range(self.gdf_onshore.shape[0]):
                    polygon = self.gdf_onshore.iloc[region_idx].geometry
                    if point.within(polygon):
                        regions_onshore[region_idx].append((point.x, point.y))
                        break

                for region_idx in range(self.gdf_offshore.shape[0]):
                    polygon = self.gdf_offshore.iloc[region_idx].geometry
                    if point.within(polygon):
                        regions_offshore[region_idx].append((point.x, point.y))
                        break

        return regions_onshore, regions_offshore

    def _create_era5_region_data(self, regions_onshore, regions_offshore):
        """
        Helper function that takes the average of all coordinates within a region and creates a new xarray dataset.

        :param regions_onshore: A list of all the regions and the coordinates that lies within this region.
        :param regions_offshore: A list of all the regions and the coordinates that lies within this region.
        """

        print("Creating era5 data for the regions. This process can take a very long time.")

        region_coords = regions_onshore + regions_offshore

        # Coordinates. Adding "on" and "off" to the name to avoid duplicates in the offshore and onshore region names.
        times = self.era_data["time"].values
        regions_on = self.gdf_onshore["name"].values
        for i in range(regions_on.shape[0]):
            regions_on[i] = regions_on[i] + " on"
        regions_off = self.gdf_offshore["name"].values
        for i in range(regions_off.shape[0]):
            regions_off[i] = regions_off[i] + " off"
        regions = np.concatenate((regions_on, regions_off))

        # Dimension/Coordinates sizes
        n_time = times.shape[0]
        n_regions = regions.shape[0]

        era_regions = []
        for region_idx in range(n_regions):
            print("Creating Dataset for region " + str(region_idx + 1) + " (" + str(
                len(region_coords[region_idx])) + " points) of " + str(n_regions))

            coords = region_coords[region_idx]
            x_coords = [str(x[0]) for x in coords]
            y_coords = [str(x[1]) for x in coords]

            era_regions.append(self.era_data.sel(
                x=xr.DataArray(x_coords, dims="points", coords={"points": list(range(0, len(x_coords)))}),
                y=xr.DataArray(y_coords, dims="points")).mean(dim="points"))

        era_regions_concat = xr.concat(era_regions, pd.Index(regions, name="region")).transpose("region", "time")

        era_regions_ds = xr.Dataset(
            data_vars=dict(
                height=(
                    ["region", "time"],
                    np.full((n_regions, n_time), np.repeat(era_regions_concat["height"].data, 8760).reshape(65, 8760))),
                wnd100m=(["region", "time"], era_regions_concat["wnd100m"].data),
                roughness=(["region", "time"], era_regions_concat["roughness"].data),
                influx_toa=(["region", "time"], era_regions_concat["influx_toa"].data),
                influx_direct=(["region", "time"], era_regions_concat["influx_direct"].data),
                influx_diffuse=(["region", "time"], era_regions_concat["influx_diffuse"].data),
                albedo=(["region", "time"], era_regions_concat["albedo"].data),
                temperature=(["region", "time"], era_regions_concat["temperature"].data),
                soil_temperature=(["region", "time"], era_regions_concat["soil temperature"].data),
                runoff=(["region", "time"], era_regions_concat["runoff"].data),
            ),
            coords=dict(
                region=(["region"], era_regions_concat["region"].data),
                time=(["time"], era_regions_concat["time"].data),
            ),
            attrs=dict(
                description="Era5 data with mean value of the coordinates within a region",
            ),
        )

        era_regions_ds.to_netcdf(config.paths["era5_regions"])


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
