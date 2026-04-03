from __future__ import annotations

import math
from dataclasses import dataclass
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str

# weather location
MONTREAL = Location(
    name="Montreal, QC",
    lat=45.5017,
    lon=-73.5673,
    timezone="America/Montreal",
)

# Data acquisition
def fetch_open_meteo_hourly(
    start_date: str,
    end_date: str,
    location: Location = MONTREAL,
    hourly_vars: List[str] | None = None,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "relative_humidity_2m",
            "surface_pressure",
            "precipitation",
            "cloud_cover",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": location.timezone,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", None)
    if times is None:
        raise RuntimeError(f"Open-Meteo response missing 'hourly.time'. Keys: {payload.keys()}")

    idx = pd.to_datetime(times)
    df = pd.DataFrame(index=idx)
    for k, v in hourly.items():
        if k == "time":
            continue
        df[k] = v
    df.index.name = "time_local"
    return df

# Preprocessing + features
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    df = df.reindex(full_idx)

    df = df.interpolate(limit=6)
    df = df.ffill().bfill()

    rename = {
        "temperature_2m": "T",
        "wind_speed_10m": "W",
        "wind_direction_10m": "Wd",
        "relative_humidity_2m": "RH",
        "surface_pressure": "P",
        "precipitation": "Prec",
        "cloud_cover": "Cloud",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Diurnal + seasonal features
    hour = df.index.hour.to_numpy()
    omega = 2 * math.pi / 24.0
    # TODO: Student can add whatever you like to the dataset here. Example here
    df["sin_day"] = np.sin(omega * hour)
    df["cos_day"] = np.cos(omega * hour)

    doy = df.index.dayofyear.to_numpy()
    omega_y = 2 * math.pi / 365.25
    # TODO: Student can add whatever you like to the dataset here
    
    # Seasonal cycle
    doy = df.index.dayofyear.to_numpy()
    omega_year = 2 * math.pi / 365.25
    df["sin_year"] = np.sin(omega_year * doy)
    df["cos_year"] = np.cos(omega_year * doy)
    return df

# TODO: Students you might find this function useful
def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    """
    Add lagged versions of one column.

    A lag of L means the value from L hours earlier is used as a predictor.
    """
    for L in lags:
        if L <= 0:
            continue
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

# TODO: Students you might find this function useful
def split_train_val(data: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the supervised dataset into:
    - training data: all rows except the last val_hours
    - validation data: the final val_hours rows
    """
    if len(data) <= val_hours + 10:
        raise ValueError("Not enough samples for requested validation window.")
    return data.iloc[:-val_hours].copy(), data.iloc[-val_hours:].copy()

# Least-squares solver and error metrics
def solve_least_squares_svd(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve the least-squares system A x ≈ y using singular value decomposition.

    This returns the coefficient vector x that minimizes ||A x - y||^2.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1.0 / S)
    x = Vh.T @ S_inv @ U.T @ y
    return x

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root mean square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# Dataset construction
def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the predictor dataframe by adding lagged weather variables.

    Included lags:
    - Temperature: 1, 2, 3, 6, 12, 24 hours
    - Wind speed: 1, 3, 6, 12, 24 hours
    - Relative humidity, pressure, cloud cover, precipitation: 1 hour
    """
    df_feat = df.copy()

    # Temperature lags
    df_feat = add_lags(df_feat, "T", [1, 2, 3, 6, 12, 24])

    # Wind lags
    df_feat = add_lags(df_feat, "W", [1, 3, 6, 12, 24])

    # Other weather lags
    for col in ["RH", "P", "Cloud", "Prec"]:
        if col in df_feat.columns:
            df_feat = add_lags(df_feat, col, [1])

    return df_feat

def prepare_supervised_data(df_feat: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    For a chosen horizon h, the target is the temperature h hours ahead:
        T_target(k) = T(k + h)

    Rows containing missing values are dropped because lagging and shifting
    naturally create NaNs near the beginning and end of the dataset.
    """
    data = df_feat.copy()

    # Future target
    data["T_target"] = data["T"].shift(-horizon)

    # Use every column except the target itself as an input feature.
    feature_cols = [
        col for col in data.columns
        if col not in ["T_target"]
    ]
    # Remove rows with missing values caused by lagging and target shifting.
    data = data.dropna()

    return data, feature_cols


# Model training and validation
def run_horizon_model(df: pd.DataFrame, horizon: int, val_hours: int = 24 * 7) -> dict:
    """
    Train and evaluate a least-squares model for one forecast horizon.

    A separate model is fitted for each horizon because the target temperature
    changes depending on how far ahead the prediction is made.
    """
    data, feature_cols = prepare_supervised_data(df, horizon)

    # Split chronologically into training and validation subsets.
    train_df, val_df = split_train_val(data, val_hours=val_hours)

    # Build input matrices and target vectors.
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["T_target"].to_numpy(dtype=float)

    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["T_target"].to_numpy(dtype=float)

    # Add a column of ones so the linear model includes an intercept term.
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_val = np.column_stack([np.ones(len(X_val)), X_val])

    # Estimate least-squares regression coefficients from the training set.
    coeffs = solve_least_squares_svd(X_train, y_train)

    # Compute model predictions on both training and validation sets.
    y_train_pred = X_train @ coeffs
    y_val_pred = X_val @ coeffs

    # Persistence baseline: predict future temperature as current temperature
    y_val_base = val_df["T"].to_numpy(dtype=float)

    # Store fitted parameters, metrics, and validation predictions.
    results = {
        "horizon_h": horizon,
        "coeffs": coeffs,
        "feature_cols": ["intercept"] + feature_cols,
        "train_rmse": rmse(y_train, y_train_pred),
        "train_mae": mae(y_train, y_train_pred),
        "val_rmse": rmse(y_val, y_val_pred),
        "val_mae": mae(y_val, y_val_pred),
        "baseline_rmse": rmse(y_val, y_val_base),
        "baseline_mae": mae(y_val, y_val_base),
        "val_index": val_df.index,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_val_base": y_val_base,
    }

    return results

# Plotting
def plot_validation(results: dict, save_dir: str = "plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    h = results["horizon_h"]
    idx = results["val_index"]

    plt.figure(figsize=(10, 4))
    plt.plot(idx, results["y_val"], label="Observed")
    plt.plot(idx, results["y_val_pred"], label="Least-squares model")
    plt.plot(idx, results["y_val_base"], label="Persistence baseline", linestyle="--")
    plt.title(f"Temperature prediction: horizon = {h} h")
    plt.ylabel("Temperature [°C]")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"validation_h{h}.png"), dpi=200)
    plt.show()

# Main program
if __name__ == "__main__":
    # Define the time period of interest.
    start_date = "2026-01-01"
    end_date = "2026-03-07"

    print(f"Fetching Open-Meteo hourly data for {MONTREAL.name}...")
    df_raw = fetch_open_meteo_hourly(start_date, end_date, location=MONTREAL)

    #print(df_raw)
    print("Preprocessing...")
    df = preprocess(df_raw)
    #print(df)

    # Build lagged predictors used by the regression model.
    print("Building lagged features...")
    df_feat = build_feature_dataframe(df)

    os.makedirs("plots", exist_ok=True)

    # Plot the temperature history over the selected time range.
    plt.figure(figsize=(10, 4))
    df["T"].plot(linewidth=1)
    plt.title("Montreal hourly temperature (2m)")
    plt.ylabel("T [°C]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/montreal_temperature.png", dpi=200)
    plt.show()

    # Forecast horizons to be tested, in hours.
    horizons = [1, 3, 6, 12, 24, 48]
    summary = []

    # Fit one least-squares model for each forecast horizon.
    for h in horizons:
        print(f"\nRunning horizon = {h} h")
        results = run_horizon_model(df_feat, horizon=h, val_hours=24 * 7)

        # Save the estimated model parameters for this horizon.
        param_df = pd.DataFrame({
            "parameter": results["feature_cols"],
            "value": results["coeffs"]
        })

        print(f"\nEstimated model parameters for horizon = {h} h:")
        print(param_df.to_string(index=False))

        param_df.to_csv(f"plots/model_parameters_h{h}.csv", index=False)

        # Store performance metrics for later summary comparison.
        summary.append({
            "horizon_h": results["horizon_h"],
            "train_rmse": results["train_rmse"],
            "train_mae": results["train_mae"],
            "val_rmse": results["val_rmse"],
            "val_mae": results["val_mae"],
            "baseline_rmse": results["baseline_rmse"],
            "baseline_mae": results["baseline_mae"],
        })

        # Plot validation predictions for this horizon.
        plot_validation(results, save_dir="plots")

    # Create and save a summary table comparing all horizons.
    summary_df = pd.DataFrame(summary)
    print("\nSummary of model performance:")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("plots/model_summary.csv", index=False)