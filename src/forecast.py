from src.forecast_data import *
from src.metrics import *
import config

import pandas as pd
from ngboost import NGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from ngboost.distns import Normal
from ngboost.scores import LogScore


class Forecast:
    """
    The class provides functions to make predictions. Predictions are made for all regions and the configured quantiles.
    """

    def __init__(self, _quantiles=None):
        """
        Initializes the required data set and the quantiles for which the predictions are to be made.

        :param _quantiles: Quantiles for which the prediction is to be made. If no quantiles are defined, they are loaded from the configuration.
        """
        self.data = ForecastData()
        self.quantiles = None

        if _quantiles is None:
            self.quantiles = config.default_quantiles
        else:
            self.quantiles = _quantiles

    def forecast_regression(self, test_size=0.25, random_state=42):
        """
        The function makes a regression with NGBoost. A separate model is trained for each region and energy source. The data is split into training and test data. The training is terminated prematurely if the score on the test data does not improve anymore.
        The quantile predictions are saved as .csv files.

        :param test_size: Ratio between training and test data
        :param random_state: Random state to allow comparability of results
        """

        results = {}
        new_columns = {}  # Helper to avoid fragmentation of the result Dataframe
        for q in self.quantiles:
            results[q] = pd.DataFrame()
            results[q]["snapshot"] = self.data.capfacts["snapshot"]
            new_columns[q] = {}

        capfacts_cols = self.data.capfacts.columns.values[1:]

        i = 1
        for col_name in capfacts_cols:
            print("Processing \"", col_name, "\" (", i, "/", len(capfacts_cols), ")")
            i += 1

            region_name, energy_type = self.data.parse_capfac_col(col_name)

            if (energy_type == EnergyType.NOT_DEFINED) or (energy_type == EnergyType.ROR):
                print("Skipped column: ", col_name)
                print("------------------------------------------------------------------\n")
            else:
                print("Create Trainings data for region: ", region_name, " with energy type: ", energy_type)

                Y, X = self.data.get_training_data(col_name)
                X_pred = self.data.shape_multi_feature_data(X)
                X_train, X_test, Y_train, Y_test = train_test_split(X_pred, Y, test_size=test_size,
                                                                    random_state=random_state)

                print("Fit Regression Model for region ", region_name)
                ngb = NGBRegressor(Dist=Normal, Score=LogScore, n_estimators=1000, random_state=42, verbose=True)
                ngb.fit(X=X_train, Y=Y_train, X_val=X_test, Y_val=Y_test, early_stopping_rounds=2)
                print("Iteration with best validation score: ", ngb.best_val_loss_itr)
                print("Feature Importances: ", ngb.feature_importances_)

                print("Predict capacity factors for region ", region_name, "with quantiles: ", self.quantiles)
                Y_dists = ngb.pred_dist(X_pred, max_iter=ngb.best_val_loss_itr)

                for q in self.quantiles:
                    Y_pred = Y_dists.ppf(q)
                    new_columns[q][col_name] = pd.Series(Y_pred)
                    self._calculate_scores(Y, Y_pred, q, Y_dists, False)
                print("------------------------------------------------------------------\n")

        for q in self.quantiles:
            new_columns[q] = pd.DataFrame(new_columns[q], index=results[q].index)
            results[q] = pd.concat([results[q], new_columns[q]], axis=1)
            self._clip_and_save(results[q], q)

    def _calculate_scores(self, Y_true, Y_pred, q, Y_dists=None, clipped=False):
        """
        Calculates and prints the scores of the predictions

        :param Y_true: Ground truth of the prediction
        :param Y_pred: Predicted values
        :param q: quantil
        :param Y_dists: Predicted distribution. NLL ist calculated if this is not None
        :param clipped: True if Y_pred is clipped, false otherwise
        """
        scores = {}
        if not clipped:
            if q == 0.5:
                scores["RMSE"] = root_mean_squared_error(Y_true, Y_pred)
                scores["MAE"] = mean_absolute_error(Y_true, Y_pred)
            else:
                s = "PL " + str(q)
                scores[s] = pinball_loss(Y_true, Y_pred, q)
            if Y_dists is not None:
                scores["NLL"] = negative_log_likelihood(Y_true, Y_dists)
        else:
            if q == 0.5:
                scores["RMSE clipped"] = root_mean_squared_error(Y_true, Y_pred)
                scores["MAE clipped"] = mean_absolute_error(Y_true, Y_pred)
            else:
                s = "PL " + str(q) + " clipped"
                scores[s] = pinball_loss(Y_true, Y_pred, q)
        print("Scores for q =", q, "\n", scores)

    def _clip_and_save(self, result, q):
        """
        Clips saves the prediction results

        :param result: prediction results for a single quantil
        :param quantil: quantil for the file name
        """
        output_dir = Path(config.result_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = "capfacts_pred_q" + str(int(q * 100)) + ".csv"
        result.to_csv(output_dir / output_file)
        print("Finished regression for q = ", q, ". Saved results to: ", output_dir / output_file)

        cols_num = result.select_dtypes(np.number).columns
        result[cols_num] = result[cols_num].clip(lower=0, upper=1.02)

        output_file = "capfacts_pred_q" + str(int(q * 100)) + "_clipped.csv"
        result.to_csv(output_dir / output_file)
        print("Finished regression for q = ", q, ". Saved the clipped results to: ", output_dir / output_file)

    def forecast_regression_grid_search(self, param_grid: dict, test_size=0.25, random_state=42, n_jobs=4, cv=5):
        """
        The function makes a regression with NGBoost. A separate model is trained for each region and energy source.
        The data is split into training and test data. The training is terminated prematurely if the score on the test
        data does not improve anymore.The quantile predictions are saved as .csv files.
        The best hyperparameters of the regression are determined from the given parameter grid using a GridSearch
        cross-validation.

        :param param_grid: Dictionary of hyperparameters
        :param test_size: Ratio between training and test data
        :param random_state: Random state to allow comparability of results
        :param cv: n-fold cross-validation
        :param n_jobs: number of parallel jobs used for fitting the grid search
        """

        results = {}
        new_columns = {}  # Helper to avoid fragmentation of the result Dataframe
        for q in self.quantiles:
            results[q] = pd.DataFrame()
            results[q]["snapshot"] = self.data.capfacts["snapshot"]
            new_columns[q] = {}

        capfacts_cols = self.data.capfacts.columns.values[1:]

        i = 1
        for col_name in capfacts_cols:
            print("Processing \"", col_name, "\" (", i, "/", len(capfacts_cols), ")")
            i += 1

            region_name, energy_type = self.data.parse_capfac_col(col_name)

            if (energy_type == EnergyType.NOT_DEFINED) or (energy_type == EnergyType.ROR):
                print("Skipped column: ", col_name)
                print("------------------------------------------------------------------\n")
            else:
                print("Create Trainings data for region: ", region_name, " with energy type: ", energy_type)

                Y, X = self.data.get_training_data(col_name)
                X_pred = self.data.shape_multi_feature_data(X)
                X_train, X_test, Y_train, Y_test = train_test_split(X_pred, Y, test_size=test_size,
                                                                    random_state=random_state)

                print("Determine best Parameters with GridSearchCV for region ", region_name)
                ngb = NGBRegressor(Dist=Normal, Score=LogScore, random_state=42, verbose=False)
                grid_search = GridSearchCV(ngb, param_grid=param_grid, n_jobs=n_jobs, cv=cv)
                grid_search.fit(X_train, Y_train)

                # cv_result = pd.DataFrame(grid_search.cv_results_)
                # print(cv_result)
                ngb_best = NGBRegressor(Dist=Normal, Score=LogScore, random_state=42, verbose=False,
                                        **grid_search.best_params_)
                print("Best estimator:", ngb_best)
                ngb_best.fit(X_train, Y_train)

                print("Predict capacity factors for region ", region_name, "with quantiles: ", self.quantiles)
                Y_dists = ngb_best.pred_dist(X_pred)

                for q in self.quantiles:
                    Y_pred = Y_dists.ppf(q)
                    new_columns[q][col_name] = pd.Series(Y_pred)
                print("------------------------------------------------------------------\n")

        for q in self.quantiles:
            new_columns[q] = pd.DataFrame(new_columns[q], index=results[q].index)
            results[q] = pd.concat([results[q], new_columns[q]], axis=1)
            self._clip_and_save(results[q], q)
