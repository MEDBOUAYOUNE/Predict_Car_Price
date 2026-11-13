"""Gradient descent for the linear regression."""

import numpy as np

from carprice.config import DEFAULT_ITERATIONS, DEFAULT_LEARNING_RATE


def calc_predictions(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """
    Step 2: predict the price of every example.

    Args:
        x (np.ndarray): Feature values (mileage), one per example.
        theta0 (float): Current intercept.
        theta1 (float): Current slope.

    Returns:
        np.ndarray: Predictions y_hat_i = theta0 + theta1 * x_i.
    """
    return theta0 + theta1 * x


def calc_errors(
        x: np.ndarray,
        y: np.ndarray,
        theta0: float,
        theta1: float
        ) -> np.ndarray:
    """
    Step 3: compute the error of every prediction.

    Args:
        x (np.ndarray): Feature values (mileage).
        y (np.ndarray): Actual prices.
        theta0 (float): Current intercept.
        theta1 (float): Current slope.

    Returns:
        np.ndarray: Errors e_i = y_hat_i - y_i.
    """
    return calc_predictions(x, theta0, theta1) - y


def calc_gradients(errors: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Step 4: compute the gradients of theta0 and theta1.

    Args:
        errors (np.ndarray): Errors e_i of every example.
        x (np.ndarray): Feature values (mileage).

    Returns:
        tuple[float, float]: (gradient_theta0, gradient_theta1), where
            gradient_theta0 = (1/m) * sum(e_i)
            gradient_theta1 = (1/m) * sum(e_i * x_i)
    """
    m = len(x)
    gradient0 = errors.sum() / m
    gradient1 = (errors * x).sum() / m
    return gradient0, gradient1


def gradient_descent(
        km: np.ndarray,
        price: np.ndarray,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        iterations: int = DEFAULT_ITERATIONS
        ) -> tuple[float, float]:
    """
    Find theta0 (intercept) and theta1 (slope) with gradient descent.

    The mileage is normalized to the 0-1 range during training so that
    the learning rate works, and the thetas are converted back to real
    km units before being returned.

    Args:
        km (np.ndarray): Mileage of every car.
        price (np.ndarray): Price of every car.
        learning_rate (float): Step size (alpha).
        iterations (int): Number of update steps.

    Returns:
        tuple[float, float]: (theta0, theta1) in real units, so that
            price = theta0 + theta1 * km.
    """
    km_min, km_max = km.min(), km.max()
    x = (km - km_min) / (km_max - km_min)

    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        errors = calc_errors(x, price, theta0, theta1)
        gradient0, gradient1 = calc_gradients(errors, x)

        tmp0 = learning_rate * gradient0
        tmp1 = learning_rate * gradient1
        theta0 -= tmp0
        theta1 -= tmp1

    theta1_real = theta1 / (km_max - km_min)
    theta0_real = theta0 - theta1_real * km_min
    return theta0_real, theta1_real
