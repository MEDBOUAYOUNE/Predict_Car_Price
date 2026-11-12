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


if __name__ == "__main__":
    df = load_dataset("./data.csv")
    if df is None:
        raise SystemExit(1)
    var = calc_var(df)
    co_var = calc_covar(df)
    slope = calc_slope(co_var=co_var, var=var)
    intercept = calc_intercept(df, slope)
    print(f"Slope: {slope}, intercept: {intercept}")
