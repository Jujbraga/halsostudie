import pandas as pd

import src.functions as f
import src.viz as v

class HealthAnalyzer:
    """
    Analyze and visualize a health dataset.

    Provides descriptive statistics, confidence intervals, hypothesis testing, power simulation,
    linear regression and plotting.

    """
    def __init__(self, df):
        self.df = df
        self.df["bmi"] = self.df["weight"] / (self.df["height"] / 100) ** 2

    # Statistics
    def calculate_avg(self):
        """Return mean, median, min and max for the numeric columns."""
        avg = self.df[["age", "height", "weight", "systolic_bp", "cholesterol"]].agg(
    ["mean", "median", "min", "max"]).round(2)
        return avg
    
    def correlation(self):
        """Return the correlation matrix, with sex and smoker encoded as 0/1."""
        df_copy = self.df.copy()
        d = df_copy.drop(columns="id")
        d["sex"] = (d["sex"] == "M").astype(int)
        d["smoker"] = (d["smoker"] == "Yes").astype(int)
        return d.corr().round(2)

    def disease_simulation(self):
        """Simulate the proportion of patients with disease."""
        return f.simulate_disease_proportion(self.df["disease"])
    
    def confidence_intervals(self, column: str="systolic_bp"):
        """Calculate the confidence interval with t-distribution and bootstrap"""
        t_low, t_high = f.calculate_ci_t(self.df[column])
        b_low, b_high = f.calculate_ci_boot(self.df[column])
        return pd.DataFrame({
            "Nedre gräns": [t_low, b_low],
            "Övre gräns": [t_high, b_high],
            "Bredd": [t_high - t_low, b_high - b_low],
        }, index=["t-fördelning", "Bootstrap"]).round(3)

    def _smoker_groups(self):
        no = self.df.loc[self.df.smoker == "No", "systolic_bp"].values
        yes = self.df.loc[self.df.smoker == "Yes", "systolic_bp"].values
        return no, yes

    def hypothesis_test(self):
        """Test whether systolic BP differs between smokers and non-smokers."""
        no, yes = self._smoker_groups()
        return f.calculate_hypothesis(no, yes)

    def power(self, rng, diffs):
        """Estimate test power for each true difference (mmHg) in `diffs`."""
        no, yes = self._smoker_groups()
        return pd.DataFrame({
            "Verklig skillnad (mmHg)": diffs,
            "Power": [f.simulate_power(no, yes, d, rng) for d in diffs],
        })

    def linear_regression(self, x, y, pred: float=None, unit: str=None):
        """Fit a simple or multiple linear regression.

        Args:
            x: Column name (simple) or list of column names (multiple).
            y: Name of the target column.
            pred: Optional values to predict for, one per x column.
            unit: Optional unit label shown in the prediction text.
        """
        cols = [x] if isinstance(x, str) else list(x)
        x = self.df[cols].to_numpy()
        y = self.df[y].to_numpy()

        return f.linear_regr(x, y, pred, unit)

    # Graphics
    def plot(self, kind: str):
        """kind: 'weight', 'cholesterol' eller 'bmi'."""
        plots = {
            "weight": v.boxplot_weight_gender,
            "cholesterol": v.scatter_cholesterol,
            "bmi": v.hist_bmi,
        }
        if kind not in plots:
            raise ValueError(f"Choose one of: {list(plots)}")
        plots[kind](self.df)

    def plot_regression(self, x_col: str, y_col: str, interc: float, coef: float):
        """Plot `y_col` vs `x_col` with a regression line from `interc` and `coef`."""
        x = self.df[[x_col]].to_numpy()
        y = self.df[y_col].to_numpy()

        return v.scatter_regression(x, y, interc, coef)

    def plot_residuals(self, results: dict):
        """Plot residuals vs predicted values from a regression result."""
        return v.scatter_residuals(results["y_pred"], results["residuals"])