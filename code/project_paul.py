from __future__ import annotations

import math
from dataclasses import dataclass
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


MONTREAL = Location(
    name="Montreal, QC",
    lat=45.5017,
    lon=-73.5673,
    timezone="America/Montreal",
)

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

    return df

# TODO: Students you might find this function useful
def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for L in lags:
        if L <= 0:
            continue
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

# TODO: Students you might find this function useful
def split_train_val(data: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) <= val_hours + 10:
        raise ValueError("Not enough samples for requested validation window.")
    return data.iloc[:-val_hours].copy(), data.iloc[-val_hours:].copy()


if __name__ == "__main__":
    start_date = "2026-01-01"
    end_date = "2026-03-07"
    
    montreal = Location(
        name="Montreal, QC",
        lat=45.5017,
        lon=-73.5673,
        timezone="America/Montreal",
    )

    print(f"Fetching Open-Meteo hourly data for {montreal.name}...")
    df_raw = fetch_open_meteo_hourly(start_date, end_date, location=montreal)
    #print(df_raw)
    print("Preprocessing...")
    df = preprocess(df_raw)
    #print(df)
    #plot initial data
    plt.figure()
    df["T"].plot(linewidth=1)
    plt.title("Montreal hourly temperature (2m)")
    plt.ylabel("T [°C]")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('plots/montreal.png')
    plt.show()

    n = df.shape[0] #get dataframe size

    #add lags for Temperature
    df = add_lags(df, 'T', [1, 2, 3])
    df['T_lag1'][0] = df['T'][0]
    df['T_lag2'][0] = df['T'][0]
    df['T_lag2'][1] = df['T'][1]
    df['T_lag3'][0] = df['T'][0]
    df['T_lag3'][1] = df['T'][1]
    df['T_lag3'][2] = df['T'][2]
    #split dataframe into training and control data
    df_train = df[:n//2].copy()
    df_control = df[n//2+1:].copy()
    #print to verify
    print(df_train.shape)
    print(df_control.shape)
    #write an Ax=b problem
    y = np.asarray(df_train['T']) #what we try to predict
    print(df_train)
    params = np.asarray(df_train[df_train.columns.drop('T')]) #prediction parameters
    print(y)
    print(params)
    print(y[0])
    A = params
    #solve the least square problem with SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    sigma_inv = np.diag(np.power(S, -1))
    x = Vh.T@sigma_inv@U.T@y
    #print(x)
    #apply it to the train data
    y_fit = A@x
    #compute errors
    rmse = np.sqrt(np.mean((y_fit - y)**2))
    mae = np.mean(np.abs(y_fit - y))
    #plot results
    plt.plot(y, label='Measurements', color='blue')
    plt.plot(y_fit, label=f'Model, RMSE={rmse}, MAE={mae}', color='red')
    plt.legend()
    plt.ylabel('Air Temperature (2m) [°C]')
    plt.xlabel('Time')
    plt.title('Air temperature measurement and prediction for Montréal, QC')
    plt.savefig('plots/leastsquares.png')
    plt.grid(True)
    plt.show()
    #minimum of n//2 and n//2+1
    m = min(df_train.shape[0], df_control.shape[0])
    #define new A matrix
    params_prediction = np.asarray(df_control[df_control.columns.drop('T')])[0:m,:]
    A_pred = params_prediction
    y_pred = np.asarray(df_control['T'])[0:m]
    #apply model to control data
    y_pred_fit = A_pred@x
    y_pred_fit = y_pred_fit[0:m]
    #compute new error
    rmse_pred = np.sqrt(np.mean((y_pred_fit - y_pred)**2))
    mae_pred = np.mean(np.abs(y_pred_fit - y_pred))
    #plot everything
    plt.plot(y_pred, label='Measurements', color='blue')
    plt.plot(y_pred_fit, label=f'Model, RMSE={rmse_pred}, MAE={mae_pred}', color='red')
    plt.legend()
    plt.ylabel('Air Temperature (2m) [°C]')
    plt.xlabel('Time')
    plt.title('Air temperature measurement and prediction for Montréal, QC')
    plt.savefig('plots/leastsquares_prediction.png')
    plt.grid(True)
    plt.show()