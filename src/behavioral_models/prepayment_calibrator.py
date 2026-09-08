import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from src.behavioral_models.prepayment_model import MortgagePrepaymentModel

class HybridPrepaymentCalibrator:
    """
    Two-Step Calibration Engine.
    1. Fits the foundational S-Curve using SciPy non-linear least squares.
    2. Trains a SciKit-Learn Random Forest on the residual errors to capture FICO/Burnout nuances.
    """
    def __init__(self, base_cpr: float):
        self.base_cpr = base_cpr

    def _sigmoid_function(self, incentive: np.ndarray, max_cpr: float,
                          steepness: float, shift_bps: float) -> np.ndarray:
        x = -steepness * (incentive - shift_bps)
        x = np.clip(x, -500, 500)
        return self.base_cpr + (max_cpr - self.base_cpr) / (1.0 + np.exp(x))
    
    def aggregate_empirical_curve(self, tape: pd.DataFrame) -> pd.DataFrame:
        df = tape.copy()
        df['incentive'] = df['contractual_rate'] - df['market_rate']
        df['incentive_bucket'] = np.round(df['incentive'] * 400) / 400

        def calculate_bucket_cpr(group):
            if group['upb_beginning'].sum() == 0:
                return 0.0
            smm = np.average(group['empirical_smm'], weights=group['upb_beginning'])
            return 1.0 - (1.0 - smm)**12
        
        bucketed = df.groupby('incentive_bucket')[['upb_beginning', 'empirical_smm']].apply(calculate_bucket_cpr).reset_index()
        bucketed.columns = ['incentive', 'cpr']
        return bucketed
    
    def fit_model(self, tape_with_smm: pd.DataFrame) -> MortgagePrepaymentModel:
        # Step 1: Calibrate teh Anchor S-Curve
        bucketed_data = self.aggregate_empirical_curve(tape_with_smm)

        lower_bound_cpr = max(self.base_cpr, 0.001)
        initial_guess_cpr = max(self.base_cpr + 0.10, 0.40)

        popt, _ = curve_fit(
            self._sigmoid_function,
            bucketed_data['incentive'].values,
            bucketed_data['cpr'].values,
            p0=[initial_guess_cpr, 400.0, 0.01],
            bounds=([lower_bound_cpr, 0.0, -0.05], [1.0, 2000.0, 0.10])
        )
        fitted_max, fitted_steepness, fitted_shift = popt

        # Step 2: Calculate Residuals (The ML Target)
        df = tape_with_smm.copy()
        df['incentive'] = df['contractual_rate'] - df['market_rate']

        # Calculate what the S_Curve predicts for every single loan
        df['s_curve_prediction'] = np.where(
            df['incentive'] <= 0.0,
            self.base_cpr,
            self._sigmoid_function(df['incentive'].values, fitted_max, fitted_steepness, fitted_shift)
        )

        # Residual = Actual Empirical CPR - S-Curve Predicted CPR
        df['residual'] = df['empirical_cpr'] - df['s_curve_prediction']

        # Step 3: Train the Machine Learning Modifier
        features = df[['incentive', 'fico', 'burnout']].values
        target = df['residual'].values
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_model.fit(features, target)

        return MortgagePrepaymentModel(
            base_cpr=self.base_cpr,
            max_cpr=fitted_max,
            steepness=fitted_steepness,
            shift_bps=fitted_shift,
            ml_residual_model=rf_model
        )
