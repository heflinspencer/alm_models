import numpy as np
from scipy.interpolate import interp1d

class YieldCurve:
    """
    A foundational class to represent an interest rate yield curve,
    capable of interpolating rates and calculating continuous discount factors.
    """
    def __init__(self, tenors: list[float], rates: list[float]):
        """
        :param tenors: Time to maturity in years (e.g., 0.5 for 6 months)
        :param rates: Corresponding spot interest rate as decimals (e.g, 0.05 for 5%)
        """
        if len(tenors) != len(rates):
            raise ValueError("Tenors and rates arrays must be of the same length.")
        
        self.tenors = np.array(tenors)
        self.rates = np.array(rates)

        # I use linear interpolation for simplicity, allowing extrapolation
        # for cash flows that fall beyond my longest tenor.
        self._interpolator = interp1d(
            self.tenors,
            self.rates,
            kind='linear',
            fill_value='extrapolate'
        )

    def get_rate(self, t: float) -> float:
        """Returns the interpolated spot rate for time t (in years)."""
        # Ensure I don't return negative time rates
        if t < 0:
            raise ValueError("Time 't' cannot be negative.")
        return float(self._interpolator(t))
    
    def get_discount_factor(self, t: float) -> float:
        """
        Calculates the continuous discount factor for time t.
        Formula: DF(t) = exp(-r * t)
        """
        if t == 0:
            return 1.0
        rate = self.get_rate(t)
        return np.exp(-rate * t)