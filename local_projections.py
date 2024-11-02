import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from statistics import NormalDist


class LocalProjectionsModel:
    def __init__(self, data, dependent_var, shock_var, market_var):
        self.data = data
        self.dependent_var = dependent_var
        self.shock_var = shock_var
        self.market_var = market_var
        self.control_vars = None
        self.results = {}
        self.robust = False  # Set to True if robust regression is needed
        self.results = None

    def construct_local_projection_data(self, event_dates, num_market_lags):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data should be a pandas DataFrame.")

        self.data["date"] = pd.to_datetime(self.data.index)
        self.data["D_t"] = 0
        self.data.loc[self.data["date"].isin(pd.to_datetime(event_dates)), "D_t"] = 1

        self.data["D_t"] = self.data["D_t"]
        self.data["R_mkt_t"] = self.data[self.market_var]
        self.control_vars = ["R_mkt_t"]

        for l in range(1, num_market_lags + 1):
            # self.data[f'R_i_t-{lag}'] = self.data[self.dependent_var].shift(lag)
            self.data["R_mkt_t" + str(l) + "lag"] = self.data[self.market_var].shift(l)
            # self.control_vars += [f'R_i_t-{lag}',f'R_mkt_t-{lag}']
            self.control_vars += ["R_mkt_t" + str(l) + "lag"]

        self.data.dropna(inplace=True)

    def time_series_local_projections(
        self,
        data,
        response,
        shock,
        exog,
        horizon,
        lags,
        newey_lags=4,
        ci_width=0.95,
        method="long_differneces",
    ):
        ## Illegal inputs
        if (ci_width >= 1) | (ci_width <= 0):
            raise NotImplementedError("CI Width must be within (0, 1), non-inclusive!")
        if horizon < 1:
            raise NotImplementedError("Estimation horizon for IRFs must be at least 1")
        if lags < 1:
            raise NotImplementedError("Number of lags in the model must be at least 1")
        ## Preliminaries
        self.ci_width = ci_width
        col_output = [
            "Shock",
            "Horizon",
            "Mean_IRF",
            "LB_IRF",
            "UB_IRF",
            "Mean_resid",
            "LB_resid",
            "UB_resid",
        ]  # Column names of the output dataframe
        irf_full = pd.DataFrame(
            columns=col_output
        )  # Empty output dataframe to be filled over every iteration
        res = pd.DataFrame()
        z_val = NormalDist().inv_cdf(
            (1 + ci_width) / 2
        )  # Determines what multiplier to use when calculating UB & LB from SE
        ## Check ordering of response variable in the full list of Y
        ## Generate copy of data for horizon h + first difference RHS variables + transform response variable to desired horizon
        for h in range(horizon + 1):
            d = data.copy()
            d[response + "forward"] = d[response].shift(
                -h
            )  # forward; equivalent to F`h'. in Stata
            ## Generate lags of RHS variables (only the first, either l0 or l1 will be used in the IRFs)
            list_RHS_forReg = [shock] + exog

            if method == "log_difference":
                d[response] = d[response] - d[response].shift(1)  # first difference
                d[response + "forward"] - d[response].shift(1)
            for l in range(1, lags + 1):
                d[response + str(l) + "lag"] = d[response].shift(
                    l
                )  # for lagged dependent variable, we will use _l1 to generate the IRF
                list_RHS_forReg = list_RHS_forReg + [response + str(l) + "lag"]

            d = d.dropna(
                axis=0
            )  # clear all rows with NAs from the lag / forward transformations
            eqn = response + "forward" + "~" + "+".join(list_RHS_forReg)
            mod = smf.ols(eqn, data=d)
            est = mod.fit(cov_type="HAC", cov_kwds={"maxlags": newey_lags})
            beta = est.params
            se = est.bse
            res[h] = est.resid
            irf = pd.DataFrame(
                [[1] * len(col_output)], columns=col_output
            )  # double list = single row
            irf["Horizon"] = h
            irf["Shock"] = shock
            irf["Mean_IRF"] = beta[shock]
            irf["LB_IRF"] = beta[shock] - z_val * se[shock]
            irf["UB_IRF"] = beta[shock] + z_val * se[shock]
            irf["Mean_resid"] = est.resid.mean()
            irf["LB_resid"] = est.resid.mean() - z_val * est.resid.std()
            irf["UB_resid"] = est.resid.mean() + z_val * est.resid.std()
            irf_full = pd.concat([irf_full, irf], axis=0)  # top to bottom concat
        ## Sort by response, shock, horizon
        irf_full = irf_full.sort_values(
            by=["Shock", "Horizon"], axis=0, ascending=[True, True]
        )
        self.residuals = res
        self.irf = irf_full
        return irf_full

    def run_local_projections(
        self, event_dates, horizon, num_lags, opt_ci=0.95, exog=False
    ):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data should be a pandas DataFrame.")

        # self.data.index = pd.to_datetime(self.data.index)
        self.data["D_t"] = 0
        self.data.loc[
            self.data.index.astype(str).str[:10].isin([ev[:10] for ev in event_dates]),
            "D_t",
        ] = 1
        self.data["R_mkt_t"] = self.data[self.market_var]

        irf = self.time_series_local_projections(
            data=self.data,  # input dataframe
            response="log_returns",  # variables in the model
            shock="D_t",  # variables whose IRFs should be estimated
            exog=self.control_vars if exog else [],
            horizon=horizon,  # estimation horizon of IRFs
            lags=num_lags,  # lags in the model
            newey_lags=2,  # maximum lags when estimating Newey-West standard errors
            ci_width=opt_ci,  # width of confidence band
        )
        self.results = irf
        self.horizons = list(range(1 + horizon))
        self.coefs = irf["Mean_IRF"]

        self.lower_bounds = irf["LB_IRF"]
        self.upper_bounds = irf["UB_IRF"]

        return irf

    def validate_stationarity(self):
        result = adfuller(self.data[self.dependent_var])
        return result[1] <= 0.05

    def validate_autocorrelation(self):
        y = self.data[self.dependent_var]
        X = self.data[[self.shock_var] + self.control_vars]
        model = (sm.RLM if self.robust else sm.OLS)(y, sm.add_constant(X)).fit()
        _, pval, _, _ = acorr_breusch_godfrey(model, nlags=1)
        return pval > 0.05

    def validate_heteroskedasticity(self):
        y = self.data[self.dependent_var]
        X = self.data[[self.shock_var] + self.control_vars]
        model = (sm.RLM if self.robust else sm.OLS)(y, sm.add_constant(X)).fit()
        _, pval = het_white(model.resid, model.model.exog)[:2]
        return pval > 0.05

    def validate_assumptions(self):
        stationarity = self.validate_stationarity()
        autocorrelation = self.validate_autocorrelation()
        heteroskedasticity = self.validate_heteroskedasticity()
        return {
            "stationarity": stationarity,
            "autocorrelation": autocorrelation,
            "heteroskedasticity": heteroskedasticity,
        }

    def plot_projections(self, residuals=True):
        if self.irf is None or self.irf.empty:
            raise ValueError("No results found. Please run local projections first.")

        # Ensure IRF columns are numeric and handle missing values
        required_columns = ["Horizon", "Mean_IRF", "LB_IRF", "UB_IRF"]
        for col in required_columns:
            if col not in self.irf.columns:
                raise ValueError(f"Column '{col}' not found in IRF data.")
            # Convert to numeric, forcing errors to NaN
            self.irf[col] = pd.to_numeric(self.irf[col], errors="coerce")
            # Drop rows with NaN in these columns
            # Ensure numeric data and handle NaNs for plotting
            self.irf["Horizon"] = pd.to_numeric(self.irf["Horizon"], errors="coerce")
            self.irf["Mean_IRF"] = pd.to_numeric(self.irf["Mean_IRF"], errors="coerce")
            self.irf["LB_IRF"] = pd.to_numeric(self.irf["LB_IRF"], errors="coerce")
            self.irf["UB_IRF"] = pd.to_numeric(self.irf["UB_IRF"], errors="coerce")
            self.irf = self.irf.dropna(subset=required_columns)

        # Plot 1: Impulse Response Function (IRF) for Coefficients (beta values)
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.irf["Horizon"],
            self.irf["Mean_IRF"],
            marker="o",
            linestyle="-",
            color="b",
            label="Estimated Coefficient",
        )
        plt.fill_between(
            self.irf["Horizon"],
            self.irf["LB_IRF"],
            self.irf["UB_IRF"],
            color="lightblue",
            alpha=0.5,
            label=f"{int(100 * self.ci_width)}% Confidence Interval",
        )
        plt.axhline(0, color="r", linestyle="--")
        plt.title(f"Impulse Response of {self.dependent_var} to {self.shock_var}")
        plt.xlabel("Horizon")
        plt.ylabel("Estimated Coefficient")
        plt.xticks(self.irf["Horizon"].unique())
        plt.grid()
        plt.legend()
        plt.show()

        # Plot 2: Residuals across Horizons (if requested)
        if residuals and hasattr(self, "residuals") and not self.residuals.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.irf["Horizon"],
                self.irf["Mean_resid"],
                marker="o",
                linestyle="-",
                color="b",
                label="Residuals",
            )
            plt.fill_between(
                self.irf["Horizon"],
                self.irf["LB_resid"],
                self.irf["UB_resid"],
                color="lightblue",
                alpha=0.5,
                label=f"{int(100 * self.ci_width)}% Confidence Interval",
            )
            plt.axhline(0, color="r", linestyle="--")
            plt.title(
                f"Residuals from the local projections of {self.dependent_var} to {self.shock_var}"
            )
            plt.xlabel("Horizon")
            plt.ylabel("Residuals")
            plt.xticks(self.irf["Horizon"].unique())
            plt.grid()
            plt.legend()
            plt.show()
