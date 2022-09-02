import config
from src.forecast import *
from src.daytime_checker import *
from src.era5_mapper import *

if __name__ == '__main__':

    # If the reduced era5 dataset is missing, create it from the full dataset
    era5_regions_path = Path(config.paths["era5_regions"])
    if era5_regions_path.is_file():
        print(f'The file {config.paths["era5_regions"]} exists')
    else:
        print(f'The file {config.paths["era5_regions"]} does not exist yet. Start creation process ...')
        mapper = Era5Mapper()
        mapper.create_era5_region()

    # Make a forecast
    forecaster = Forecast()
    forecaster.forecast_regression(test_size=config.test_size, random_state=config.random_state)

    # Uses a GridSearchCV to find the best parametrization for the regression.
    # forecaster.forecast_regression_grid_search(config.param_grid)

    # Not integrated in the workflow, but allows to check for daytime
    # checker = DaytimeChecker()
    # lon, lat = checker.get_centroid_cea(True, "DE0 0")
    # checker.is_daytime(lon, lat)

