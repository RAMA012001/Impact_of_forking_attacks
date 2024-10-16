import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class LocalProjections:
    def __init__(self, data, outcome, event_var, controls=None, event_time=None):
        """
        Initialize the Local Projections model.

        Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        outcome (str): The name of the dependent variable (outcome of interest).
        event_var (str): The name of the variable indicating the event.
        controls (list): List of control variable names.
        event_time (str): The column indicating event time.
        """
        self.data = data
        self.outcome = outcome
        self.event_var = event_var
        self.controls = controls if controls is not None else []
        self.event_time = event_time
        self.irfs = None
        self.horizons = None

    def fit(self, max_horizon=10):
        """
        Fit the local projections model for each horizon.

        Parameters:
        max_horizon (int): The maximum number of horizons to estimate the response.

        Returns:
        A dictionary containing impulse response functions (IRFs) for each horizon.
        """
        irfs = {}
        self.horizons = np.arange(max_horizon + 1)

        for h in self.horizons:
            # Shift the dependent variable to create the horizon-specific outcome
            self.data[f"{self.outcome}_h{h}"] = self.data.groupby(self.event_time)[
                self.outcome
            ].shift(-h)

            # Perform the regression for each horizon
            y = self.data[f"{self.outcome}_h{h}"]
            X = pd.get_dummies(self.data[self.event_var], drop_first=True)

            if self.controls:
                X = pd.concat([X, self.data[self.controls]], axis=1)

            X = sm.add_constant(X, has_constant="add")
            model = sm.OLS(y, X, missing="drop")
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": h})

            irfs[h] = results.params

        self.irfs = irfs
        return irfs

    def bootstrap_irfs(self, num_bootstraps=1000, max_horizon=10, seed=None):
        """
        Perform bootstrapping to generate confidence intervals for the IRFs.

        Parameters:
        num_bootstraps (int): Number of bootstrap samples.
        max_horizon (int): Maximum number of horizons.
        seed (int): Seed for reproducibility.

        Returns:
        A dictionary containing bootstrap confidence intervals for each horizon.
        """
        np.random.seed(seed)
        bootstrap_irfs = {h: [] for h in range(max_horizon + 1)}

        for _ in range(num_bootstraps):
            irfs = self.fit(max_horizon=max_horizon)

            for h in self.horizons:
                bootstrap_irfs[h].append(irfs[h])

        ci_lower = {
            h: np.percentile(bootstrap_irfs[h], 2.5, axis=0) for h in self.horizons
        }
        ci_upper = {
            h: np.percentile(bootstrap_irfs[h], 97.5, axis=0) for h in self.horizons
        }

        return ci_lower, ci_upper

    def plot_irfs(self, ci_lower=None, ci_upper=None):
        """
        Plot the impulse response functions with optional confidence intervals.

        Parameters:
        ci_lower (dict): Lower bound of the confidence intervals.
        ci_upper (dict): Upper bound of the confidence intervals.
        """
        plt.figure(figsize=(10, 6))

        for h in self.horizons:
            plt.plot(
                self.horizons,
                [self.irfs[h].get(self.event_var, 0) for h in self.horizons],
                label=f"Horizon {h}",
            )

        if ci_lower and ci_upper:
            for h in self.horizons:
                plt.fill_between(
                    self.horizons,
                    [ci_lower[h].get(self.event_var, 0) for h in self.horizons],
                    [ci_upper[h].get(self.event_var, 0) for h in self.horizons],
                    color="gray",
                    alpha=0.3,
                )

        plt.title("Impulse Response Functions (IRFs)")
        plt.xlabel("Horizon")
        plt.ylabel("Response")
        plt.legend()
        plt.show()

    def summary(self):
        """
        Print a summary of the fitted model's IRFs.
        """
        if self.irfs:
            for h in self.horizons:
                print(f"Horizon {h}:", self.irfs[h])
        else:
            print("No IRFs available. Run the `fit()` method first.")
