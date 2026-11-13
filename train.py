from carprice.config import DATA_PATH
from carprice.data import load_dataset
from carprice.gradient import gradient_descent
from carprice.model import save_thetas


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    if df is None:
        raise SystemExit(1)
    theta0, theta1 = gradient_descent(
        df["km"].to_numpy(dtype=float),
        df["price"].to_numpy(dtype=float),
    )
    save_thetas(theta0, theta1)
    print(f"Intercept (theta0): {theta0}, slope (theta1): {theta1}")
