<div align="center">

# 🚗 Predict Car Price

### Linear regression from scratch: predict a car's price from its mileage

**No scikit-learn. No `polyfit`. Just gradient descent, NumPy and math.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20loading-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-plots-11557c)
![ML](https://img.shields.io/badge/Machine%20Learning-from%20scratch-orange)

</div>

---

## 📖 Table of contents

- [Overview](#-overview)
- [How it works](#-how-it-works)
- [Project structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Measuring precision](#-measuring-precision)
- [Design decisions](#-design-decisions)
- [Limitations](#-limitations)
- [Cheat sheet](#-cheat-sheet)

---

## 🎯 Overview

Given the mileage of a car, this project estimates its price. It is built in two programs, as machine learning projects usually are:

| Program | File | Job |
|---|---|---|
| **Training** | `train.py` | Reads `data.csv`, learns the best line with gradient descent, saves the result to `thetas.json` |
| **Prediction** | `predict.py` | Asks for a mileage, reads `thetas.json`, prints the estimated price |

Everything else (plotting, precision metrics, a closed-form cross-check) is there to **see**, **measure** and **verify** the model.

```
$ python3 train.py
Intercept (theta0): 8477.13, slope (theta1): -0.02124

$ python3 predict.py
Enter the car mileage in km (whole number, e.g. 85000): 85000
Estimated price: 6672.13
```

---

## 🧠 How it works

### 1. The model

A straight line through the data:

```
estimatePrice(mileage) = θ0 + θ1 × mileage
```

| Symbol | Name | Meaning |
|---|---|---|
| **θ0** | intercept | The price of a car at 0 km |
| **θ1** | slope | How much the price changes for each extra km (negative: more km, lower price) |

Before training, both start at **0**, so the untrained model predicts a price of 0.

### 2. Training with gradient descent

Gradient descent starts at θ0 = θ1 = 0 and repeats the same five steps. Each pass makes the predictions a little better:

| Step | What happens | Formula |
|---|---|---|
| 1 | Start with θ0 = 0, θ1 = 0 and choose a learning rate α | |
| 2 | Predict the price of every car | `ŷᵢ = θ0 + θ1·xᵢ` |
| 3 | Measure each error | `eᵢ = ŷᵢ − yᵢ` |
| 4 | Compute the two gradients | `g0 = (1/m)·Σ eᵢ`   `g1 = (1/m)·Σ eᵢ·xᵢ` |
| 5 | Update **both** thetas at the same time | `θ0 ← θ0 − α·g0`   `θ1 ← θ1 − α·g1` |

`m` is the number of cars in the dataset.

> ⚠️ **Simultaneous update.** Both gradients are computed from the *old* thetas before either theta changes. Updating θ0 first and then using it for θ1's gradient gives a wrong result.

### 3. Why the mileage is normalized

Mileage is in the hundreds of thousands, while prices are in the thousands. On raw values, the slope gradient is enormous and training explodes unless the learning rate is absurdly small (around `1e-11`).

So during training, mileage is rescaled to the 0–1 range:

```
x = (km − km_min) / (km_max − km_min)
```

After training, the thetas are converted back to **real km units**, so `predict.py` works with ordinary mileages:

```
θ1_real = θ1 / (km_max − km_min)
θ0_real = θ0 − θ1_real × km_min
```

### 4. Saving the model

The trained values are stored in `thetas.json`:

```json
{"intercept": 8477.13, "slope": -0.02124}
```

If this file does not exist, `predict.py` warns you and falls back to θ0 = θ1 = 0.

---

## 🗂 Project structure

```
Predict_Car_Price/
├── data.csv              24 cars: km, price
├── requirements.txt      numpy, pandas, matplotlib
│
├── train.py              ▶ program 2: train and save thetas.json
├── predict.py            ▶ program 1: mileage in, price out
├── plot.py               📈 scatter of the data + the regression line
├── precision.py          🎯 train / test split and precision metrics
├── ols_check.py          ✅ closed-form cross-check of the result
│
└── carprice/             shared code (no duplication)
    ├── config.py         file paths and default learning rate / iterations
    ├── data.py           load_dataset, train_test_split
    ├── model.py          pp_price, load_thetas, save_thetas
    ├── gradient.py       predictions, errors, gradients, gradient_descent
    ├── metrics.py        MAE, MSE, RMSE, R², MAPE, precision rate, tolerance rate
    └── ols.py            variance, covariance, closed-form slope and intercept
```

The top-level scripts stay thin: they connect the pieces and handle input and output. All the formulas live in `carprice/`.

---

## ⚙️ Installation

Requires **Python 3.10 or newer**.

```bash
git clone <repo-url>
cd Predict_Car_Price

python3 -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

---

## 🚀 Usage

Run every command from the project folder.

### 1. Train the model

```bash
python3 train.py
```

Reads `data.csv`, runs gradient descent, and writes `thetas.json`.

### 2. Predict a price

```bash
python3 predict.py
```

Enter a mileage in km as a whole number. Invalid input (text, a negative number) gets a clear error message.

### 3. Plot the data and the regression line

```bash
python3 plot.py
```

Shows every car as a point and the fitted line in red. If the model has not been trained yet, only the points are shown.

### 4. Measure precision

```bash
python3 precision.py
```

### 5. Cross-check with the closed-form solution

```bash
python3 ols_check.py
```

---

## 🎯 Measuring precision

A model must be judged on cars it has **never seen**. `precision.py` shuffles the data with a fixed seed, trains on 80% of the cars, and tests on the other 20%.

Sample output:

```
Train size: 19 | Test size: 5
Intercept: 8519.17 | Slope: -0.021504
MAE:  368.00
RMSE: 419.97
R2:   0.2698
MAPE: 5.32%
Precision rate (100 - MAPE): 94.68%
Predictions within 10%: 100%
```

| Metric | What it tells you |
|---|---|
| **MAE** | Average miss in dollars |
| **RMSE** | Like MAE, but big misses count more |
| **R²** | Share of the price variation the model explains (1 = perfect, 0 = no better than the average) |
| **MAPE** | Average error as a percentage of the real price |
| **Precision rate** | `100 − MAPE`, a single easy-to-read score |
| **Within tolerance** | Percentage of predictions within 10% of the real price |

All metrics are implemented by hand in `carprice/metrics.py`, and the split is implemented in `carprice/data.py`.

> 📝 With only 24 cars, the test set has 5 rows, so these numbers move noticeably with a different random seed. Treat them as an estimate, not a guarantee.

---

## 🧩 Design decisions

- **Gradient descent, as the project requires.** `polyfit` and scikit-learn are not used in the training code.
- **A closed-form check on the side.** `ols_check.py` computes the exact least-squares answer from variance and covariance. Gradient descent should land very close to it. The two agree when training has fully converged:

  | | Gradient descent (10,000 iterations) | Closed form |
  |---|---|---|
  | Intercept | 8499.5996 | 8499.5996 |
  | Slope | -0.0214490 | -0.0214490 |

  With the default of 1,000 iterations, the result is close but not fully converged (intercept 8477.13, slope -0.02124). Raise `DEFAULT_ITERATIONS` in `carprice/config.py` for the exact fit.
- **NumPy instead of Python loops** for the math, so the code is short and fast.
- **One source of truth.** File names and default settings live in `carprice/config.py`, and each function exists once.
- **A readable model file.** `thetas.json` uses named keys, so it can be inspected and edited by hand.
- **Documented code.** Every function has a docstring with its formula, arguments and return value.

---

## ⚠️ Limitations

- **A straight line extends forever.** Beyond about 396,000 km, the model predicts a negative price. The data only covers 22,899 to 240,000 km, so anything outside that range is extrapolation.
- **One feature only.** Mileage is the only input. Age, brand and condition also affect price.
- **A small dataset.** 24 cars is enough to learn the idea, but not for a reliable real-world model.
- **Fixed learning rate.** It works because the mileage is normalized, but it is tuned for this data.

---

## 🧾 Cheat sheet

| I want to... | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Train the model | `python3 train.py` |
| Predict a price | `python3 predict.py` |
| See the data and the fitted line | `python3 plot.py` |
| Measure precision | `python3 precision.py` |
| Verify against the exact solution | `python3 ols_check.py` |
| Change the learning rate or iterations | edit `carprice/config.py` |

---

<div align="center">

Built to understand how machine learning really works, one gradient step at a time. 📉

</div>
