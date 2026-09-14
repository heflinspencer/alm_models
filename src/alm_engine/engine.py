import numpy as np
from src.alm_engine.yield_curve import YieldCurve
from src.alm_engine.instruments import Instrument

class ALMEngine:
    """
    Asset and Liability Management (ALM) Engine.
    Agggregates instruments, calculates EVE, and applies regulatory rate shocks.
    """
    def __init__(self, portfolio: list[Instrument], baseline_curve: YieldCurve):
        self.portfolio = portfolio
        self.baseline_curve = baseline_curve

    def calculate_eve(self, curve: YieldCurve = None) -> float:
        """
        Calculates the Economic Value of Equity (EVE) by summing the Present Value
        of all cash flows in the portfolio.
        """
        active_curve = curve if curve is not None else self.baseline_curve
        eve = 0.0

        for instrument in self.portfolio:
            cash_flows = instrument.get_cash_flows(active_curve)
            for t, cf in cash_flows.items():
                discount_factor = active_curve.get_discount_factor(t)
                eve += cf * discount_factor
        
        return eve
    
    def shift_curve_parallel(self, bps_shift: float, floor: float = 0.0) -> YieldCurve:
        """
        Creates a new YieldCurve by shifting the baseline rates.
        :param bps_shift: the shock in basis points (e.g. 200 for +2%, -200 for -2%)
        :param floor: The minimum allowable interest rate.
        """
        shift_decimal = bps_shift / 10000.0
        new_rates = self.baseline_curve.rates + shift_decimal

        # Apply rate floor 
        new_rates = np.maximum(new_rates, floor)

        return YieldCurve(tenors=self.baseline_curve.tenors, rates=new_rates)
    
    def shift_curve_non_parallel(self, bps_shifts: list[float], floor: float = 0.0):
        """
        Creates a new YieldCurve by applying specific shocks to individual tenors.

        :param bps_shifts: A list of shocks in basis points. MUST match the length
        and order of self.baseline_curve.tenors.

        :param floor: The minimum allowable interest rate.
        """
        if len(bps_shifts) != len(self.baseline_curve.tenors):
            raise ValueError(f"Expected {len(self.baseline_curve.tenors)} shifts, got {len(bps_shifts)}")
        
        # Convert list of bps to a decimal numpy array
        shifts_decimal = np.array(bps_shifts) / 10000.0

        # Add the specific shift to each specific tenor's rate
        new_rates = self.baseline_curve.rates + shifts_decimal

        # Apply the rate floor
        new_rates = np.maximum(new_rates, floor)

        return YieldCurve(tenors=self.baseline_curve.tenors, rates=new_rates)
    
    def calculate_eve_non_parallel_shock(self, bps_shifts: list[float]) -> dict[str, float]:
        """
        Calculates the delta EVE under a non-parallel rate shock (e.g., Steepener/Flattener).
        """
        baseline_eve = self.calculate_eve()

        shocked_curve = self.shift_curve_non_parallel(bps_shifts)
        shocked_eve = self.calculate_eve(curve=shocked_curve)

        delta_eve = shocked_eve - baseline_eve

        return {
            "baseline_eve": baseline_eve,
            "shocked_eve": shocked_eve,
            "delta_eve": delta_eve
        }

    def calculate_eve_sensitivity(self, bps_shock: float) -> dict[str, float]:
        """
        Calculates the delta EVE under a parallel rate shock.
        """
        baseline_eve = self.calculate_eve()

        shocked_curve = self.shift_curve_parallel(bps_shock)
        shocked_eve = self.calculate_eve(curve=shocked_curve)

        delta_eve = shocked_eve - baseline_eve

        return {
            "baseline_eve": baseline_eve,
            "shocked_eve": shocked_eve,
            "delta_eve": delta_eve
        }