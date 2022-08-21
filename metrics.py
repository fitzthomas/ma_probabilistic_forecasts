"""
This script includes metrics that allow calculating the quality of predictions.
"""

import numpy as np
from ngboost.distns import Normal
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import mean_squared_error as mse


def coverage_fraction(y_true, y_low, y_high) -> float:
    """
    Computes the percentage of observations that fall between the quantile predictions. Two single-valued quantile
    predictions, with quantiles 𝛼_𝑙𝑜𝑤 < 𝛼_ℎ𝑖𝑔ℎ describe a (𝛼_ℎ𝑖𝑔ℎ − 𝛼_𝑙𝑜𝑤)%-confidence interval. The
    coverage fraction value should be close to this percentage.

    :param y_true: Ground truth, correct target values.
    :param y_low: Small quantil of the estimated target values.
    :param y_high: Large quantil of the estimated target values.
    :return: Percentage of observations that fall between the quantile predictions.
    """
    return np.mean(np.logical_and(y_true >= y_low, y_true <= y_high))


def negative_log_likelihood(y_true, y_pred_dist: Normal) -> float:
    """
    Computes the negative log likelihood (NLL) of the probabilistic forecast.

    :param y_true: Ground truth, correct target values.
    :param y_pred_dist: Estimated target values in the form of a normal distributon as defined in ngboost.
    :return: The NLL is a negative floating point. Smaller values are better.
    """
    return -y_pred_dist.logpdf(y_true).mean()


def pinball_loss(y_true, y_pred, quantil=0.5) -> float:
    """
    Calculates the pinball loss of a prediction considering a bias or quantile.

    :param y_true: Ground truth, correct target values.
    :param y_pred: Estimated target values.
    :param quantil: Quantile or bias assumed in the calculation.
    :return: The pinball loss output is a non-negative floating point. The best value is 0.0.
    """
    return mean_pinball_loss(y_true, y_pred, alpha=quantil)


def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Calculates the root mean squared error (RMSE) of a prediction.
    RSME should only be used for a point prediction without bias (50% quantil)

    :param y_true: Ground truth, correct target values.
    :param y_pred: Estimated target values.
    :return: A non-negative floating point value (the best value is 0.0).
    """
    return mse(y_true, y_pred, squared=False)


def mean_absolute_error(y_true, y_pred) -> float:
    """
    Calculates the mean absolute error (MAE) of a prediction.
    MAE should only be used for a point prediction without bias (50% quantil)

    :param y_true: Ground truth, correct target values.
    :param y_pred: Estimated target values.
    :return: A non-negative floating point value (the best value is 0.0).
    """
    return mae(y_true, y_pred)
