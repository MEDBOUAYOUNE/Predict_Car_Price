"""Metrics measuring how close the predicted prices are to the real ones."""

import math

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute error (MAE).

    MAE = (1/m) * sum(|y_true_i - y_pred_i|)

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: Average size of the error, in the same unit as the price.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean squared error (MSE).

    MSE = (1/m) * sum((y_true_i - y_pred_i)^2)

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: Average squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the root mean squared error (RMSE).

    RMSE = sqrt(MSE). It penalizes large errors more than the MAE.

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: RMSE, in the same unit as the price.
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R2).

    R2 = 1 - SS_res / SS_tot, where
        SS_res = sum((y_true_i - y_pred_i)^2)
        SS_tot = sum((y_true_i - mean(y_true))^2)

    1 is a perfect fit, 0 is no better than always predicting the mean,
    and a negative value is worse than that.

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: The R2 score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def mean_absolute_percentage_error(
        y_true: np.ndarray,
        y_pred: np.ndarray
        ) -> float:
    """
    Compute the mean absolute percentage error (MAPE).

    MAPE = (100/m) * sum(|y_true_i - y_pred_i| / |y_true_i|)

    Args:
        y_true (np.ndarray): Actual prices (none of them should be 0).
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: Average error as a percentage of the real price.
    """
    assert np.all(y_true != 0), "y_true should not contain 0"
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def precision_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the precision rate of the model as a percentage.

    precision rate = 100 - MAPE (never below 0).

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.

    Returns:
        float: Precision rate, between 0 and 100.
    """
    return max(0.0, 100 - mean_absolute_percentage_error(y_true, y_pred))


def within_tolerance_rate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: float = 0.10
        ) -> float:
    """
    Compute the share of predictions close enough to the real price.

    A prediction counts as correct when
    |y_true_i - y_pred_i| <= tolerance * |y_true_i|.

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.
        tolerance (float): Allowed error as a fraction of the real price
            (0.10 means 10%).

    Returns:
        float: Percentage of predictions within the tolerance.
    """
    hits = np.abs(y_true - y_pred) <= tolerance * np.abs(y_true)
    return float(np.mean(hits) * 100)
