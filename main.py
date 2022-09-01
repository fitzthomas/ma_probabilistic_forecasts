from src.forecast import *


if __name__ == '__main__':

    era5_regions_path = Path(config.paths["era5_regions"])
    if era5_regions_path.is_file():
        print(f'The file {config.paths["era5_regions"]} exists')
    else:
        print(f'The file {config.paths["era5_regions"]} does not exist yet. Start creation process ...')
        create_era5_region()

    forecaster = Forecast()
    forecaster.forecast_regression()