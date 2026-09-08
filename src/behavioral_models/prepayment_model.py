import numpy as np

class MortgagePrepaymentModel:
    """
    Hybrid Mortgage Prepayment Model.
    Combines a deterministic S-Curve (for MRM asymptotic bounds) with a
    Machine Learning regressor (to capture behavioral residuals like FICO and Burnout).
    """
    def __init__(self, base_cpr: float, max_cpr: float,
                 steepness: float, shift_bps: float,
                 ml_residual_model=None):
        """
        :param base_cpr: Baseline turnover (demographics/life events) when rates rise.
        :param max_cpr: Maximum prepayment speed during the peak of a refinancing boom.
        :param steepness: How aggressively customers react to rate drops (beta).
        :param shift_bps: The rate incentive required to hit the midpoint of the refi boom.
        """
        self.base_cpr = base_cpr
        self.max_cpr = max_cpr
        self.steepness = steepness
        self.shift_bps = shift_bps
        self.ml_residual_model = ml_residual_model

    def calculate_cpr(self, contractual_rate: float, market_rate: float,
                      fico: int = 700, burnout: int = 0) -> float:
        # 1. S-Curve anchor
        incentive = contractual_rate - market_rate

        # If market rates are higher than the contractual rate, there is zero
        # financial incentive to refinance. Only baseline demographic turnover applies.
        if incentive <= 0.0:
            anchor_cpr = self.base_cpr
        else: 
            # S-Curve calculation for positive refinancing incentive.
            # np.clip prevents math overflow errors if the steepness/incentive gets extreme.
            x = -self.steepness * (incentive - self.shift_bps)
            x = np.clip(x, -500.0, 500)
            sigmoid = 1.0 / (1.0 + np.exp(x))
            # Scale the sigmoid to bounce between the Base CPR and the Max CPR
            anchor_cpr = self.base_cpr + (self.max_cpr - self.base_cpr) * sigmoid

        # 2. The Machine Learning modifier
        ml_adjustment = 0.0
        if self.ml_residual_model is not None:
            # Predict expects a 2D array: [[incentive, fico, burnout]]
            features = np.array([[incentive, fico, burnout]])
            ml_adjustment = self.ml_residual_model.predict(features)[0]

        # 3. Combine and Enforce MRM Physical Bounds (0%-100%)
        final_cpr = anchor_cpr + ml_adjustment
        return float(np.clip(final_cpr, 0.0, 0.99))
    
    def __repr__(self) -> str:
        ml_status = "Active" if self.ml_residual_model else "None"
        return (f"HybridPrepaymentModel(S-Curve Max={self.max_cpr:.2f}, "
                f"ML_Residuals={ml_status})")