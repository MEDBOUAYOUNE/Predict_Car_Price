"""Loading and splitting the car dataset."""

import math

import numpy as np
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the car dataset from a CSV file.

    Args:
        path (str): Path to a non-empty .csv file with the columns
            'km' (mileage) and 'price'.

    Returns:
        pd.DataFrame: The dataset, or None if the file could not be read
            (an error message is printed in that case).
    """
    assert isinstance(path, str), "Path should be str"
    assert path.endswith(".csv"), "Path should be in Csv format"
    assert len(path) > 0, "Path shouldn't be empty"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: the file '{path}' was not found. Check the path.")
    except pd.errors.EmptyDataError:
        print(f"Error: the file '{path}' is empty.")
    except ValueError as e:
        print(f"Error: the file '{path}' has an invalid format ({e}).")
    except Exception as e:
        print(f"Unexpected error while loading '{path}': {e}")


def train_test_split(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle the data and split it into a training and a test set.

    Args:
        x (np.ndarray): Feature values (mileage).
        y (np.ndarray): Target values (price).
        test_size (float): Fraction of the data used for the test set,
            strictly between 0 and 1.
        random_state (int): Seed, so the split is reproducible.

    Returns:
        tuple: (x_train, x_test, y_train, y_test).
    """
    assert len(x) == len(y), "x and y should have the same length"
    assert 0 < test_size < 1, "test_size should be between 0 and 1"
    indexes = np.random.default_rng(random_state).permutation(len(x))
    n_test = math.ceil(len(x) * test_size)
    test_idx, train_idx = indexes[:n_test], indexes[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
