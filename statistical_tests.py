import numpy as np
from scipy import stats


class TTest:
    def __init__(self):
        self.name = "T-test"

    def verify_assumptions(self, abnormal_returns):
        # Assumption 1: Normality of returns
        _, normality_p_value = stats.shapiro(abnormal_returns)
        normality_verified = "yes" if normality_p_value > 0.05 else "no"

        # Assumption 2: Independence of observations
        # This is difficult to verify directly,
        # so we might use autocorrelation as a proxy
        autocorrelation = np.corrcoef(abnormal_returns[:-1], abnormal_returns[1:])[0, 1]
        independence_verified = "yes" if abs(autocorrelation) < 0.1 else "no"

        return [
            {
                "Assumption": "Normality",
                "Verification_Method": "Shapiro-Wilk Test",
                "Verified": normality_verified,
            },
            {
                "Assumption": "Independence",
                "Verification_Method": "Autocorrelation Check",
                "Verified": independence_verified,
            },
        ]

    def calculate_test(self, abnormal_returns, std_dev):
        # Compute the t-statistic
        t_stat = abnormal_returns / std_dev
        return t_stat


class BMPTest:
    def __init__(self):
        self.name = "BMPT-test"

    def verify_assumptions(self, abnormal_returns):
        # Assumption 1: Homoscedasticity across firms
        # We check the variance of returns across different firms and compare them
        variances = [
            np.var(crypto_abnormal_returns)
            for crypto_abnormal_returns in abnormal_returns
        ]
        homoscedasticity_verified = (
            "yes" if max(variances) / min(variances) < 2 else "no"
        )

        # Assumption 2: Normally distributed abnormal returns
        normality_test_stat, normality_p_value = stats.shapiro(
            np.concatenate(abnormal_returns)
        )
        normality_verified = "yes" if normality_p_value > 0.05 else "no"

        return [
            {
                "Assumption": "Homoscedasticity",
                "Verification_Method": "Variance Comparison",
                "Verified": homoscedasticity_verified,
            },
            {
                "Assumption": "Normality",
                "Verification_Method": "Shapiro-Wilk Test",
                "Verified": normality_verified,
            },
        ]

    def calculate_test(self, standardized_abnormal_returns):
        # Compute the t-statistic for BMP Test
        t_stat = np.mean(standardized_abnormal_returns) / (
            np.std(standardized_abnormal_returns)
            / np.sqrt(len(standardized_abnormal_returns))
        )
        return t_stat


class GRANKTTest:
    def __init__(self, abnormal_returns):
        self.name = "G-Rank-test"
        self.ranks = abnormal_returns.rank()

    def verify_assumptions(self):
        # Assumption 1: No missing data
        no_missing_data_verified = "yes" if not np.isnan(self.ranks).any() else "no"

        # Assumption 2: Independence of ranked returns
        # We use the autocorrelation method again for proxy checking
        autocorrelation = np.corrcoef(self.ranks[:-1], self.ranks[1:])[0, 1]
        independence_verified = "yes" if abs(autocorrelation) < 0.1 else "no"

        return [
            {
                "Assumption": "No Missing Data",
                "Verification_Method": "NaN Check",
                "Verified": no_missing_data_verified,
            },
            {
                "Assumption": "Independence",
                "Verification_Method": "Autocorrelation Check",
                "Verified": independence_verified,
            },
        ]

    def calculate_test(self):
        # Calculate Z for GRANK-T Test
        Z = np.mean(self.ranks) / (np.std(self.ranks) / np.sqrt(len(self.ranks)))
        t_stat = Z * ((len(self.ranks) - 1) / (len(self.ranks) - Z**2))
        return t_stat


class GSignTest:
    def __init__(self, abnormal_returns):
        self.name = "G-Sign-test"
        self.positive_abnormal_returns = (abnormal_returns > 0).astype(int)

    def verify_assumptions(self):
        # Assumption 1: Data is binary (positive/negative returns)
        binary_verified = (
            "yes" if all(np.isin(self.positive_abnormal_returns, [0, 1])) else "no"
        )

        # Assumption 2: Sufficient sample size for approximation to normal
        sufficient_sample_verified = (
            "yes" if len(self.positive_abnormal_returns) > 30 else "no"
        )

        return [
            {
                "Assumption": "Binary Data",
                "Verification_Method": "Binary Check",
                "Verified": binary_verified,
            },
            {
                "Assumption": "Sufficient Sample Size",
                "Verification_Method": "Sample Size Check",
                "Verified": sufficient_sample_verified,
            },
        ]

    def calculate_test(self, p_hat=0.5):
        w = self.positive_abnormal_returns.sum()  # Count of positive abnormal returns
        N = len(self.positive_abnormal_returns)
        # Compute z-statistic for the G-Sign Test
        z_stat = (w - N * p_hat) / np.sqrt(N * p_hat * (1 - p_hat))
        return z_stat
