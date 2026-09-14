import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.behavioral_models.nmd_beta_model import NMDBetaModel

class NMDBetaCalibrator:
    """
    Asymmetric NMD Beta Calibrator.
    Regresses historical deposit rate changes against market rate changes,
    splitting the data into rising (up-cycle) and falling (down-cycle) envrionments.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def generate_synthetic_history(self, periods: int = 60) -> pd.DataFrame:
        """
        Simulates 5 years (60 months) of historical rate data.
        Injects a hidden 30% Up-beta and 70% Down-beta to test my calibrator.
        """
        time = np.linspace(0, 10, periods)
        # Create a market rate that cycles up and down
        market_rates = 0.03 + 0.02 * np.sin(time)

        deposit_rates = np.zeros(periods)
        deposit_rates[0] = 0.04 # Starting deposit rate

        for t in range(1, periods):
            delta_mkt = market_rates[t] - market_rates[t-1]

            # The bank's hidden ALM policy
            true_up_beta = 0.30
            true_down_beta = 0.70

            if delta_mkt > 0:
                delta_dep = delta_mkt * true_up_beta
            else:
                delta_dep = delta_mkt * true_down_beta

            # Add a tiny bit of idiosyncratic noise (management discretion)
            delta_dep += np.random.normal(0, 0.0005)

            deposit_rates[t] = max(0.0, deposit_rates[t-1] + delta_dep)

        return pd.DataFrame({
            'month': np.arange(periods),
            'market_rate': market_rates,
            'deposit_rate': deposit_rates
        })
    
    def calibrate_betas(self, history: pd.DataFrame) -> NMDBetaModel:
        """
        Calculates the asymmetric betas using OLS regression (forcing intercept to 0).
        """
        df = history.copy()

        # 1. Calculate month-over-month deltas
        df['delta_mkt'] = df['market_rate'].diff()
        df['delta_dep'] = df['deposit_rate'].diff()
        df = df.dropna()

        # 2. Split into Up-cycles and Down-cycles
        up_cycle = df[df['delta_mkt'] > 0]
        down_cycle = df[df['delta_mkt'] <= 0]

        # 3. Fit OLS without intercept (Beta = pure slope)
        def fit_beta(x: pd.Series, y: pd.Series) -> float:
            if len(x) == 0: return 0.0
            X = x.values.reshape(-1, 1)
            Y = y.values
            reg = LinearRegression(fit_intercept=False).fit(X,Y)
            return float(reg.coef_[0])
        
        calibrated_up_beta = fit_beta(up_cycle['delta_mkt'], up_cycle['delta_dep'])
        calibrated_down_beta = fit_beta(down_cycle['delta_mkt'], down_cycle['delta_dep'])

        # Get the most recent market rate to act as the baseline for future shocks
        base_market_rate = float(df['market_rate'].iloc[-1])

        return NMDBetaModel(
            up_beta=calibrated_up_beta,
            down_beta=calibrated_down_beta,
            base_market_rate=base_market_rate
        )
