from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from constants import Cryptocurrency


class CryptoAttack:
    def __init__(self, crypto: Cryptocurrency, attack_date: str):
        self.attack_date = pd.to_datetime(attack_date).tz_localize("UTC")
        self.crypto_name = crypto.name
        self.data = crypto.data
        self.attack_index = crypto.attack_dates.index(attack_date)
        self.attack_name = (
            "Attack " + str(self.attack_index + 1) + " on " + self.crypto_name
        )

    def calculate_CAR(
        self,
        return_type="log_returns",
        horizon=11,
        estimation_widow_size=80,
        robust=False,
    ):
        start_date = self.attack_date - timedelta(days=estimation_widow_size)
        end_date = self.attack_date + timedelta(days=horizon)
        self.data = self.data.loc[
            (self.data.index <= end_date) & (self.data.index >= start_date)
        ].ffill()
        self.data["returns"] = self.data[return_type]
        self.data["market_returns"] = self.data[f"market_{return_type}"]
        self.estimation_window_data = self.data[
            self.data.index <= self.attack_date - timedelta(days=1)
        ]
        market_returns = self.data["market_returns"].to_numpy()
        X = self.estimation_window_data["market_returns"].to_numpy()
        y = self.estimation_window_data["returns"].to_numpy()

        if robust:
            X = sm.add_constant(X)
            market_returns = sm.add_constant(
                market_returns
            )  # Add constant term to new vector
            model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            results = model.fit()
            model = results
        else:
            model = LinearRegression()
            X = X.reshape(-1, 1)
            market_returns = market_returns.reshape(-1, 1)
            model.fit(X, y)

        self.model = model
        pred = model.predict(market_returns)
        self.data["estimated_return"] = pred

        # Calculate the abnormal returns (AR)
        self.data["AR"] = self.data["returns"] - self.data["estimated_return"]
        self.attack_window_data = self.data[
            self.data.index >= self.attack_date - timedelta(days=1)
        ]

        # Calculate the cumulative abnormal return (CAR)
        self.attack_window_data["CAR"] = self.attack_window_data["AR"].cumsum()

        # Find the minimal CAR
        self.min_CAR = np.nanmin(self.attack_window_data["CAR"])

        self.CAR = self.attack_window_data.iloc[-1]["CAR"]
