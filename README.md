# Integrating Probabilistic Forecasts in Energy System Modelling

The code was written as part of my master thesis on *Integrating Probabilistic Forecasts in Energy System Modelling*.
The program allows probabilistic predictions of capacity factors based on historical weather data.

## Setup

The code has been tested on both Windows and Linux.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your operating system
2. Clone this repository and change the folder in your terminal  
   `git clone https://github.com/fitzthomas/ma_probabilistic_forecasts.git`  <br />
   `cd ma_probabilistic_forecasts`
3. Install the conda environment by typing `conda env create -f environment.yml`. If this fails, pls install the used
   libraries manually
    1. `conda create --name ma_probabilistic_forecasts python=3.10`
    2. `conda activate ma_probabilistic_forecasts`
    3. **[xarray](https://xarray.pydata.org/en/stable/getting-started-guide/installing.html)**:
       `conda install -c conda-forge xarray dask netCDF4 bottleneck`
    4. **[geopandas](https://geopandas.org/en/stable/):** `conda install -c conda-forge geopandas`
    5. **[atlite](https://atlite.readthedocs.io/en/latest/installation.html):** `conda install -c conda-forge atlite`
    6. **[jupyterlab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html):** `conda install -c conda-forge jupyterlab`
    7. **[scikit-learn](https://scikit-learn.org/stable/install.html):** `conda install -c conda-forge scikit-learn` and
       for Intel Systems
       `conda install -c intel scikit-learn`\
    8. **[ngboost](https://github.com/stanfordmlgroup/ngboost):** `conda install -c conda-forge ngboost`
    9. **[seaborn](https://seaborn.pydata.org/index.html):** `conda install -c conda-forge seaborn`
    10. **[ephem](https://pypi.org/project/ephem/):** `conda install -c anaconda ephem`
    11. **[pypsa](https://pypsa.org/):** `conda install -c conda-forge pypsa`
4. Change environment if not done in step 3 with `conda activate ma_probabilistic_forecasts`
5. Start the script by typing `python3 main.py` in terminal

### PyCharm

If you are using PyCharm, please see
the [documentation](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#15696dbb)
on how to set up the Conda environment.

## Input Data

The required input data is located in the `resources` directory. Mandatory for the prediction is

- **capfacs_37.csv**: Capacity factors
- **europe-2013-era5-regions.nc**: Historical era5 weather data set, reduced to the regions used for the prediction

Additional input data is required if the reduced weather data has not yet been provided or calculated:

- **regions_offshore_elec_s_37.geojson** and **regions_onshore_elec_s_37.geojson**: Shapefiles
- **europe-2013-era5.nc**: Full era5 weather dataset. This has to be downloaded
  manually. Available [here](https://zenodo.org/record/4709858#.YZUVdCYo8WM).

## Application Examples

### Era5Mapper

With the Era5Mapper you can create the full reduced version of the era5 dataset.

```python
mapper = Era5Mapper()
mapper.create_era5_region()
```

### Forecast

Makes a forecast for all regions. The feature selection and the calculated quantiles can be changed in `config.py`.
You can also use a set of different parameters, also defined in `config.py` to optimize the hyperparameters of the
model.

```python
forecaster = Forecast()
forecaster.forecast_regression(test_size=config.test_size, random_state=config.random_state)

# Uses a set of parameters to find the best parameters
forecaster.forecast_regression_grid_search(config.param_grid)
```

### DaytimeChecker

This class is not integrated into the workflow but allows to determine if it is day or night time in a certain region

```python
checker = DaytimeChecker()
lon, lat = checker.get_centroid_cea(True, "DE0 0")
checker.is_daytime(lon, lat)
```

## PyPSA-Eur Integration

A big thank you goes to Martha for providing a workflow for integrating capacity factor predictions into
PyPSA-Eur ([Link to the Repository](https://github.com/martacki/thomas-ma)).

## Other notes

- The jupyter notebook in the `notebooks/` directory was used to create plots for the thesis. The code was not revised
  after the migration from Notebooks to Python scripts
- Other notebooks that served as implementations and for testing during the development are located in the
  directories `notebooks_archive/` and `testing_playground`, but have been removed from the current repository in commit
  xy.