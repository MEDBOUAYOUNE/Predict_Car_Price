import pandas as pd
import os


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
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Error: the file '{path}' was not found. Check the path.")
    except pd.errors.EmptyDataError:
        print(f"Error: the file '{path}' is empty.")
    except ValueError as e:
        print(f"Error: the file '{path}' has an invalid format ({e}).")
    except Exception as e:
        print(f"Unexpected error while loading '{path}': {e}")


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
    estimated_price = intercept + (slope * mileage)
    # print(estimated_price)
    return estimated_price

if __name__ == "__main__":
    try:
        mileage = int(input("Enter the car mileage in km (whole number, e.g. 85000): "))
    except ValueError:
        print("Error: invalid mileage. Please enter a whole number like 85000.")
        raise SystemExit(1)
    if mileage < 0:
        print("Error: mileage cannot be negative.")
        raise SystemExit(1)
    # load_dataset("./data.csv")
    pp_price(mileage, 0, 0)
