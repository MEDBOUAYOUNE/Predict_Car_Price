import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from carprice.config import DATA_PATH, THETAS_PATH
from carprice.data import load_dataset
from carprice.model import load_thetas, pp_price


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
    df = load_dataset(DATA_PATH)
    if df is None:
        raise SystemExit(1)
    if os.path.exists(THETAS_PATH):
        intercept, slope = load_thetas()
        plot_data(df, intercept, slope)
    else:
        print("Model not trained yet: showing the data only.")
        plot_data(df)
