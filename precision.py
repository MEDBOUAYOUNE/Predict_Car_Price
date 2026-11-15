import pandas as pd

from carprice.config import DATA_PATH
from carprice.data import load_dataset
from carprice.metrics import precision_rate
from carprice.model import load_thetas, pp_price


def evaluate(df: pd.DataFrame) -> None:
    """
    Measure the precision of the already trained model.

    Uses the intercept and slope saved by train.py (thetas.json); nothing
    is trained again here.

    Args:
        df (pd.DataFrame): Dataset with 'km' and 'price' columns.
    """
    x = df["km"].to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)

    intercept, slope = load_thetas()
    y_pred = pp_price(x, intercept, slope)

    print(f"Precision rate (100 - MAPE): "
          f"{precision_rate(y, y_pred):.2f}%")


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    if df is None:
        raise SystemExit(1)
    evaluate(df)
