import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from pp_car import load_thetas, pp_price


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


def plot_data(
        df: pd.DataFrame,
        intercept: float | None = None,
        slope: float | None = None
        ) -> None:
    """
    Plot the price of each car against its mileage as a scatter plot,
    and the regression line on top of it when the model is given.

    Each point is one car, so every price stays paired with its own km.

    Args:
        df (pd.DataFrame): Dataset with 'km' and 'price' columns.
        intercept (float | None): Intercept of the trained model.
        slope (float | None): Slope of the trained model.
            The line is drawn only if both intercept and slope are given.
    """
    _, ax = plt.subplots()
    ax.scatter(df["km"], df["price"], label="Data")
    if intercept is not None and slope is not None:
        line_x = [df["km"].min(), df["km"].max()]
        line_y = [pp_price(x, intercept, slope) for x in line_x]
        ax.plot(line_x, line_y, color="red", label="Linear regression")
        ax.legend()
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1e3:.0f}K"))
    ax.set_xlabel("Km")
    ax.set_ylabel("Price in $")
    ax.set_title("Price Of Cars based on Mileage")
    plt.show()


if __name__ == "__main__":
    df = load_dataset("./data.csv")
    if df is None:
        raise SystemExit(1)
    if os.path.exists("thetas.json"):
        intercept, slope = load_thetas()
        plot_data(df, intercept, slope)
    else:
        print("Model not trained yet: showing the data only.")
        plot_data(df)
