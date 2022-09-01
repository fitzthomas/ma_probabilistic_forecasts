from src.era5_mapper import *

import config
import pandas as pd
import xarray as xr
from pathlib import Path


class ForecastData:
    """
    The class provides functions to load and process the era5 weather data and capacity factors.
    """

    def __init__(self):
        """
        Loads the era5 dataset and capacity factors used for the forecast
        """
        self.era5 = None
        self.capfacts = None
        self._open_input_data()

    def _open_input_data(self):
        """
        Auxiliary function to load the input data.
        """
        era5_path = Path(config.paths["era5_regions"])
        capfacts = Path(config.paths["capfacs"])

        if era5_path.is_file():
            print(f'The file {config.paths["era5_regions"]} exists. Open data ...')
            self.era5 = xr.open_dataset(filename_or_obj=config.paths["era5_regions"], engine="netcdf4")
        else:
            print(f'The file {config.paths["era5_regions"]} does not exist.')
            print("Please use the create_era5_region() in the era5_mapper to create the file.")

        if capfacts.is_file():
            print(f'The file {config.paths["capfacs"]} exists. Open data ...')
            self.capfacts = pd.read_csv(config.paths["capfacs"])
        else:
            print(f'The file {config.paths["capfacs"]} does not exist.')
            print("Please provide the necessary capacity factors.")

    def find_countries_in_capfacts(self, country_name="") -> list:
        """
        Returns the full region names and energy types of the given name abbreviation that can be found in the .csv file with capacity factors.
        :param country_name: Two character abbreviation of the searched country
        :return: list of all regions and energy types to the given country name
        """
        countries = []
        for column in self.capfacts:
            if column.find(country_name) >= 0:
                countries.append(column)
        return countries

    def parse_capfac_col(self, column_name: str) -> (str, EnergyType):
        """
        Returns a tuple of the region name and energy type for a given column name of the capfacts .csv file
        :param column_name: column name of the capfacts .csv file
        :return: Tuple of a region name and energy type, None if no region is found
        """
        col_args = column_name.split(" ")
        if len(col_args) == 3:
            region_name = col_args[0]
            energy_type = EnergyType.get_energy_type(col_args[2])
            return region_name, energy_type
        return None, None

    def get_training_data(self, column_name: str) -> (np.ndarray, dict):
        """
        Creates and returns the the training data set with the relevant data for a given column name from the capfacts .csv file.
        The training data set is a tuple of a numpy array of capacity factors (target values) and a dictionary of the era5 data (feature data),
        :param column_name: column name of the capfacts .csv file
        :return: Tuple of capacity factor (Y) and trainings data (X)
                X                       : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
                Y                       : DataFrame object or List or numpy array of outcomes (n) in Numeric format.
        """
        region_name, energy_type = self.parse_capfac_col(column_name)
        era5_region_name = get_era5_region_name(region_name, energy_type)
        features = config.feature_set.get(energy_type)

        Y = self.capfacts[column_name].values
        X = {}
        for feature in features:
            X[feature] = self.era5.sel(region=era5_region_name)[feature.value].values

        return Y, X

    def shape_multi_feature_data(self, training_data: dict):
        """
        Reshapes the trainingsdata in an array of shape (n_samples, n_features)
        (8760, 2) ---> [[x_f1_1, x_f2_1], [x_f1_2, x_f2_2], ... , [x_f1_8760, x_f2_8760]]
        :param training_data as a dictinary of multiple 1-d arrays:
        :return: trainingsdata in array of shape (n_samples,  n_features)
        """
        arrays = list(training_data. values())
        return np.stack(arrays, axis=-1)


