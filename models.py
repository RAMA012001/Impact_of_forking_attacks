import pandas as pd
import numpy as np
from datetime import timedelta
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


class MarketModel:
    def __init__(
        self,
        asset_returns,
        market_returns,
        event_date,
        estimation_window_size,
        horizon,
        robust=False,
    ):
        assert asset_returns.index.equals(
            market_returns.index
        ), "The indices of the two series do not match!"
        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.alpha = None
        self.beta = None
        self.event_date = pd.to_datetime(event_date)
        self.estimation_window_size = estimation_window_size
        self.horizon = horizon
        self.robust = robust

        data = pd.DataFrame(
            data={"asset_returns": asset_returns, "market_returns": market_returns}
        )
        data.index = data.index.tz_localize(None)
        start_date = self.event_date - timedelta(days=self.estimation_window_size)
        end_date = self.event_date + timedelta(days=horizon)
        self.data = data.loc[
            (data.index <= end_date) & (data.index >= start_date)
        ].ffill()

    def verify_assumptions(self):
        self.estimation_window_data = self.data[
            self.data.index <= self.event_date - timedelta(days=1)
        ]
        assumptions = [
            {
                "name": "Linearity between asset and market returns",
                "verified": False,
                "method": "OLS regression to check significance of beta.",
            },
            {
                "name": "Homoscedasticity",
                "verified": False,
                "method": "Breusch-Pagan test.",
            },
            {
                "name": "No autocorrelation of errors",
                "verified": False,
                "method": "Durbin-Watson test.",
            },
        ]

        # Perform either OLS or Robust Regression based on self.robust
        X = sm.add_constant(self.estimation_window_data["market_returns"])

        if self.robust:
            # Use Robust Linear Model (Huber regression)
            model = sm.RLM(
                self.estimation_window_data["asset_returns"],
                X,
                M=sm.robust.norms.HuberT(),
            ).fit()
        else:
            # Use Ordinary Least Squares (OLS) Regression
            model = sm.OLS(self.estimation_window_data["asset_returns"], X).fit()

        self.alpha, self.beta = model.params

        # Check if beta is significant (p-value < 0.05) only if using OLS,
        # as robust regression does not provide p-values.
        if not self.robust and model.pvalues[1] < 0.05:
            assumptions[0]["verified"] = True

        # Breusch-Pagan test for homoscedasticity (only meaningful for OLS)
        if not self.robust:
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            if bp_test[1] > 0.05:
                assumptions[1]["verified"] = True

        # Durbin-Watson test for autocorrelation
        dw_test = sm.stats.stattools.durbin_watson(model.resid)
        if 1.5 < dw_test < 2.5:
            assumptions[2]["verified"] = True

        return assumptions

    def calculate_CAR(self):
        self.event_window_data = self.data[
            self.data.index >= self.event_date - timedelta(days=1)
        ]
        expected_returns = (
            self.alpha + self.beta * self.event_window_data["market_returns"]
        )
        self.abnormal_returns = (
            self.event_window_data["asset_returns"] - expected_returns
        )
        CAR = np.sum(self.abnormal_returns)
        return CAR


class MarketAdjustedModel:
    def __init__(
        self, asset_returns, market_returns, event_date, estimation_window_size, horizon
    ):
        assert asset_returns.index.equals(
            market_returns.index
        ), "The indices of the two series do not match!"
        self.asset_returns = asset_returns
        self.market_returns = market_returns
        self.event_date = pd.to_datetime(event_date)
        self.estimation_window_size = estimation_window_size
        self.horizon = horizon

        data = pd.DataFrame(
            data={"asset_returns": asset_returns, "market_returns": market_returns}
        )
        data.index = data.index.tz_localize(None)
        start_date = self.event_date - timedelta(days=self.estimation_window_size)
        end_date = self.event_date + timedelta(days=horizon)
        self.data = data.loc[
            (data.index <= end_date) & (data.index >= start_date)
        ].ffill()

    def verify_assumptions(self):
        self.estimation_window_data = self.data[
            self.data.index <= self.event_date - timedelta(days=1)
        ]
        assumptions = [
            {
                "name": "Asset returns move identically with the market",
                "verified": False,
                "method": "Check correlation between asset and market returns.",
            }
        ]

        # Check if correlation is close to 1
        correlation = np.corrcoef(
            self.estimation_window_data["asset_returns"],
            self.estimation_window_data["market_returns"],
        )[0, 1]
        if np.abs(correlation - 1) < 0.1:
            assumptions[0]["verified"] = True

        return assumptions

    def calculate_CAR(self):
        self.event_window_data = self.data[
            self.data.index >= self.event_date - timedelta(days=1)
        ]
        expected_returns = self.event_window_data["market_returns"]
        self.abnormal_returns = (
            self.event_window_data["asset_returns"] - expected_returns
        )
        CAR = np.sum(self.abnormal_returns)
        return CAR


class MeanAdjustedModel:
    def __init__(
        self, asset_returns, peer_returns, event_date, estimation_window_size, horizon
    ):
        assert asset_returns.index.equals(
            peer_returns.index
        ), "The indices of the two series do not match!"
        self.asset_returns = asset_returns
        self.peer_returns = peer_returns
        self.mean_peer_return = np.mean(peer_returns)
        self.event_date = pd.to_datetime(event_date)
        self.estimation_window_size = estimation_window_size
        self.horizon = horizon

        data = pd.DataFrame(
            data={"asset_returns": asset_returns, "market_returns": peer_returns}
        )
        data.index = data.index.tz_localize(None)
        start_date = self.event_date - timedelta(days=self.estimation_window_size)
        end_date = self.event_date + timedelta(days=horizon)
        self.data = data.loc[
            (data.index <= end_date) & (data.index >= start_date)
        ].ffill()

    def verify_assumptions(self):
        assumptions = [
            {
                "name": "Comparable assets are representative",
                "verified": False,
                "method": (
                    "Check similarity (e.g., correlation)"
                    "between asset and peer group returns."
                ),
            }
        ]

        # Check if correlation with peer group is significant
        self.estimation_window_data = self.data[
            self.data.index <= self.event_date - timedelta(days=1)
        ]
        assumptions = [
            {
                "name": "Asset returns move identically with the market",
                "verified": False,
                "method": "Check correlation between asset and market returns.",
            }
        ]

        # Check if correlation is close to 1
        correlation = np.corrcoef(
            self.estimation_window_data["asset_returns"],
            self.estimation_window_data["peer_returns"],
        )[0, 1]
        if np.abs(correlation) > 0.5:
            assumptions[0]["verified"] = True
        return assumptions

    def calculate_CAR(self):
        self.event_window_data = self.data[
            self.data.index >= self.event_date - timedelta(days=1)
        ]
        expected_returns = self.mean_peer_return
        self.abnormal_returns = (
            self.event_window_data["asset_returns"] - expected_returns
        )
        CAR = np.sum(self.abnormal_returns)
        return CAR
