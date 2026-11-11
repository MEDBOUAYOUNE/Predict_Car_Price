
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
    pp_price(mileage, 0, 0)
