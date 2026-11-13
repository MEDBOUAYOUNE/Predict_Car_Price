"""The linear model: prediction, and saving / loading its parameters."""

import json

from carprice.config import THETAS_PATH


def load_thetas(path: str = THETAS_PATH) -> tuple[float, float]:
    """
    Read the trained intercept (theta0) and slope (theta1) from a JSON file.

    If the file does not exist (the model has not been trained yet), both
    values are 0, as required before training.

    Args:
        path (str): Path to the JSON file written by the training program.

    Returns:
        tuple[float, float]: (intercept, slope).
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return float(data["intercept"]), float(data["slope"])
    except FileNotFoundError:
        print(f"Warning: '{path}' not found, the model is not trained. "
              "Using intercept = 0 and slope = 0.")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        print(f"Error: '{path}' is invalid. "
              "Using intercept = 0 and slope = 0.")
    return 0.0, 0.0


def save_thetas(theta0: float, theta1: float, path: str = THETAS_PATH) -> None:
    """
    Save the trained parameters to a JSON file for the prediction program.

    Args:
        theta0 (float): Intercept.
        theta1 (float): Slope.
        path (str): Destination file.
    """
    with open(path, "w") as f:
        json.dump({"intercept": theta0, "slope": theta1}, f)


def pp_price(
        mileage: int,
        intercept: float = 0,
        slope: float = 0
        ) -> float:
    """
    Predict the price of a car from its mileage.

    Uses a linear model: estimated_price = intercept + (slope * mileage),
    where intercept is the price at 0 km and slope is the price change
    per km. Both default to 0 (model not trained).

    Args:
        mileage (int): Mileage of the car in km.
        intercept (float): Intercept of the model.
        slope (float): Slope of the model.

    Returns:
        float: Predicted price.
    """
    return intercept + (slope * mileage)
