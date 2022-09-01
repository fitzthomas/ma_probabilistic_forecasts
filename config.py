from src.energy_type import EnergyType
from src.features import Feature

from sklearn.tree import DecisionTreeRegressor

"""
The configuration file contains all global data, such as file paths.
"""

resource_path = "resources/"
result_path = "results"
paths = {
    "era5_eu_2013": resource_path + "europe-2013-era5.nc",
    "era5_tutorial": resource_path + "europe-2013-era5-tutorial.nc",
    "offshore_shape": resource_path + "regions_offshore_elec_s_37.geojson",
    "onshore_shape": resource_path + "regions_onshore_elec_s_37.geojson",
    "capfacs": resource_path + "capfacs_37.csv",
    "era5_regions": resource_path + "europe-2013-era5-regions.nc"
}

"""
Determines which features are selected to calculate the capacity factor of a certain energy type.
"""
feature_set = {
    EnergyType.OFFWIND_AC: [Feature.HEIGHT, Feature.WND100M, Feature.ROUGHNESS],
    EnergyType.OFFWIND_DC: [Feature.HEIGHT, Feature.WND100M, Feature.ROUGHNESS],
    EnergyType.ONWIND: [Feature.HEIGHT, Feature.WND100M, Feature.ROUGHNESS],
    EnergyType.SOLAR: [Feature.INFLUX_TOA, Feature.INFLUX_DIRECT, Feature.INFLUX_DIFFUSE, Feature.TEMPERATURE],
    EnergyType.ROR: []
}

"""
Default quantiles that are used for the prediction if not defined otherwise
"""
default_quantiles = [0.4, 0.5, 0.6]

"""
Default setting for the NGBoost regression
"""
test_size = 0.25
random_state = 42

"""
Example of a parameter grid for GridSearchCV hyperparameter optimization
"""
base1 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=2)
base2 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)
base3 = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4)
param_grid= {
    'Base': [base1, base2, base3],
    'n_estimators': [500, 100, 1000],
    'learning_rate': [0.01],
    'minibatch_frac': [1, 0.5],
    'col_sample': [1, 0.5]
}