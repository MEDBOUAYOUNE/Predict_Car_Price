import pandas as pd
import numpy as np

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

# First let's set our X and Y for our y = a +bx --> Y(P_price) = A(Price for Km 0) + B(Price Decressed On Each added Km)X(Km)
# FOR VAR --> f(x) : In Our data_set X in KM --> 

def calc_var(df : pd.DataFrame) -> float:
    """
    """

    print(df.info())
    mean = df['km'].mean()
    total = 0
    print(mean)
    for _km in df['km']:
        total += (_km - mean)**2
    n = df["km"].to_numpy().sum()
    var = total*(1/n)
    print(var)
    return var

# For CoVar --> f(X,Y) = total(Xi - MeanX)*total(Yi - MeanY) * 1/n
def calc_covar(df : pd.DataFrame) -> float:
    """
    """
    mean_x = df['km'].mean()
    mean_y = df['price'].mean()

    total = 0

    for _km, _price in zip(df["km"], df["price"]):
        total += (_km - mean_x)*(_price - mean_y)

    n = df["km"].to_numpy().sum()
    covar = total * (1 / (n))
    print(total)
    return total

        

if __name__ == "__main__":
    df = load_dataset("./data.csv")
    var = calc_var(df)
    co_var = calc_covar(df)
