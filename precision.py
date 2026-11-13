import pandas as pd

from carprice.config import DATA_PATH
from carprice.data import load_dataset, train_test_split
from carprice.gradient import gradient_descent
from carprice.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_rate,
    r2_score,
    root_mean_squared_error,
    within_tolerance_rate,
)
from carprice.model import pp_price


def evaluate(df: pd.DataFrame) -> None:
    """
    Train the model on a training set and measure its precision on a
    test set it has never seen.

    Args:
        df (pd.DataFrame): Dataset with 'km' and 'price' columns.
    """
    x = df["km"].to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    intercept, slope = gradient_descent(x_train, y_train)
    y_pred = pp_price(x_test, intercept, slope)

    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")
    print(f"Intercept: {intercept:.2f} | Slope: {slope:.6f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2:   {r2_score(y_test, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2f}%")
    print(f"Precision rate (100 - MAPE): "
          f"{precision_rate(y_test, y_pred):.2f}%")
    print(f"Predictions within 10%: "
          f"{within_tolerance_rate(y_test, y_pred):.0f}%")


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    if df is None:
        raise SystemExit(1)
    evaluate(df)
