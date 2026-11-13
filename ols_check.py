from carprice.config import DATA_PATH
from carprice.data import load_dataset
from carprice.ols import calc_covar, calc_intercept, calc_slope, calc_var


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    if df is None:
        raise SystemExit(1)
    var = calc_var(df)
    co_var = calc_covar(df)
    slope = calc_slope(co_var=co_var, var=var)
    intercept = calc_intercept(df, slope)
    print(f"Slope: {slope}, intercept: {intercept}")
