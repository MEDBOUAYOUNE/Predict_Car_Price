"""Closed-form least-squares check for the car price model.

This is NOT the training method. It computes the exact slope and
intercept with the variance / covariance formulas so the result of
gradient descent can be compared against it.
"""

import numpy as np
import pandas as pd


def calc_var(df: pd.DataFrame) -> float:
    """
    Compute the variance of the mileage.

    Var(X) = (1/n) * sum((x_i - mean_x)^2)

    Args:
        df (pd.DataFrame): Dataset with a 'km' column.

    Returns:
        float: Variance of the mileage.
    """
    x = df["km"].to_numpy(dtype=float)
    return float(np.mean((x - x.mean()) ** 2))


def calc_covar(df: pd.DataFrame) -> float:
    """
    Compute the covariance between mileage and price.

    Cov(X, Y) = (1/n) * sum((x_i - mean_x) * (y_i - mean_y))

    Args:
        df (pd.DataFrame): Dataset with 'km' and 'price' columns.

    Returns:
        float: Covariance of mileage and price.
    """
    x = df["km"].to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def calc_slope(co_var: float, var: float) -> float:
    """
    Compute the slope (theta1): the price change for each added km.

    slope = Cov(X, Y) / Var(X)

    Args:
        co_var (float): Covariance of mileage and price.
        var (float): Variance of the mileage.

    Returns:
        float: The slope.
    """
    return co_var / var


def calc_intercept(df: pd.DataFrame, slope: float) -> float:
    """
    Compute the intercept (theta0): the price of a car at 0 km.

    The regression line passes through the point (mean_x, mean_y), so
    intercept = mean_y - slope * mean_x

    Args:
        df (pd.DataFrame): Dataset with 'km' and 'price' columns.
        slope (float): The slope of the line.

    Returns:
        float: The intercept.
    """
    return df["price"].mean() - slope * df["km"].mean()
