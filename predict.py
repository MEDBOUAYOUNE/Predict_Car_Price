from carprice.config import THETAS_PATH
from carprice.model import load_thetas, pp_price


if __name__ == "__main__":
    try:
        mileage = int(input("Enter the car mileage in km (whole number, e.g. 85000): "))
    except ValueError:
        print("Error: invalid mileage. Please enter a whole number like 85000.")
        raise SystemExit(1)
    if mileage < 0:
        print("Error: mileage cannot be negative.")
        raise SystemExit(1)
    intercept, slope = load_thetas(THETAS_PATH)
    print(f"Estimated price: {pp_price(mileage, intercept, slope):.2f}")
