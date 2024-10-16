import numpy as np
from scipy import stats


class TTest:
    def verify_assumptions(self, returns):
        # Assumption 1: Normality of returns
        _, normality_p_value = stats.shapiro(returns)
        normality_verified = "yes" if normality_p_value > 0.05 else "no"

        # Assumption 2: Independence of observations
        # This is difficult to verify directly,
        # so we might use autocorrelation as a proxy
        autocorrelation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        independence_verified = "yes" if abs(autocorrelation) < 0.1 else "no"

        return [
            {
                "Assumption": "Normality",
                "Verification Method": "Shapiro-Wilk Test",
                "Verified": normality_verified,
            },
            {
                "Assumption": "Independence",
                "Verification Method": "Autocorrelation Check",
                "Verified": independence_verified,
            },
        ]

    def calculate_test(self, abnormal_returns, std_dev):
        # Compute the t-statistic
        t_stat = abnormal_returns / std_dev
        return t_stat


class BMPTest:
    def verify_assumptions(self, returns):
        # Assumption 1: Homoscedasticity across firms
        # We check the variance of returns across different firms and compare them
        variances = [np.var(firm_returns) for firm_returns in returns]
        homoscedasticity_verified = (
            "yes" if max(variances) / min(variances) < 2 else "no"
        )

        # Assumption 2: Normally distributed abnormal returns
        normality_test_stat, normality_p_value = stats.shapiro(np.concatenate(returns))
        normality_verified = "yes" if normality_p_value > 0.05 else "no"

        return [
            {
                "Assumption": "Homoscedasticity",
                "Verification Method": "Variance Comparison",
                "Verified": homoscedasticity_verified,
            },
            {
                "Assumption": "Normality",
                "Verification Method": "Shapiro-Wilk Test",
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
    def verify_assumptions(self, ranks):
        # Assumption 1: No missing data
        no_missing_data_verified = "yes" if not np.isnan(ranks).any() else "no"

        # Assumption 2: Independence of ranked returns
        # We use the autocorrelation method again for proxy checking
        autocorrelation = np.corrcoef(ranks[:-1], ranks[1:])[0, 1]
        independence_verified = "yes" if abs(autocorrelation) < 0.1 else "no"

        return [
            {
                "Assumption": "No Missing Data",
                "Verification Method": "NaN Check",
                "Verified": no_missing_data_verified,
            },
            {
                "Assumption": "Independence",
                "Verification Method": "Autocorrelation Check",
                "Verified": independence_verified,
            },
        ]

    def calculate_test(self, ranked_abnormal_returns):
        # Calculate Z for GRANK-T Test
        Z = np.mean(ranked_abnormal_returns) / (
            np.std(ranked_abnormal_returns) / np.sqrt(len(ranked_abnormal_returns))
        )
        t_stat = Z * (
            (len(ranked_abnormal_returns) - 1) / (len(ranked_abnormal_returns) - Z**2)
        )
        return t_stat


class GSignTest:
    def verify_assumptions(self, positive_abnormal_returns):
        # Assumption 1: Data is binary (positive/negative returns)
        binary_verified = (
            "yes" if all(np.isin(positive_abnormal_returns, [0, 1])) else "no"
        )

        # Assumption 2: Sufficient sample size for approximation to normal
        sufficient_sample_verified = (
            "yes" if len(positive_abnormal_returns) > 30 else "no"
        )

        return [
            {
                "Assumption": "Binary Data",
                "Verification Method": "Binary Check",
                "Verified": binary_verified,
            },
            {
                "Assumption": "Sufficient Sample Size",
                "Verification Method": "Sample Size Check",
                "Verified": sufficient_sample_verified,
            },
        ]

    def calculate_test(self, w, N, p_hat):
        # Compute z-statistic for the G-Sign Test
        z_stat = (w - N * p_hat) / np.sqrt(N * p_hat * (1 - p_hat))
        return z_stat
