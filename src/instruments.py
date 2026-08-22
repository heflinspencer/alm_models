from abc import ABC, abstractmethod
import numpy as np
from src.yield_curve import YieldCurve

class Instrument(ABC):
    """
    Abstract base class for all balance sheet instruments.
    Forces all subclasses to implement the get_cash_flows method.
    """
    def __init__(self, notional: float, rate: float, maturity: float):
        """
        :param notional: The principal amount (positive for assets, negative for liabilities.)
        :param rate: The fixed annual interest rate (e.g., 0.04 for 4%).
        :param maturity: Time to maturity in years.
        """
        self.notional = notional
        self.rate = rate
        self.maturity = maturity

    @abstractmethod
    def get_cash_flows(self) -> dict[float, float]:
        """
        Returns a dictionary mapping time t (in years) to its respective cash flow amount.
        """
        pass

class TermDeposit(Instrument):
    """
    A liability product that pays principal and compounded interest
    in a single bullet payment at maturity.
    """
    def get_cash_flows(self) -> dict[float, float]:
        if self.maturity <= 0:
            return {0.0: self.notional}
        
        # Total payoff at maturity using annual compounding
        payoff = self.notional * ((1 + self.rate) ** self.maturity)
        return {self.maturity: payoff}
    
class RetailMortgage(Instrument):
    """
    An asset product that pays fixed annual installments, amortising the principal
    down to zero over the maturity of the loan.
    """
    def get_cash_flows(self) -> dict[float, float]:
        periods = int(np.floor(self.maturity))

        if periods <= 0 or self.rate == 0.0:
            # Fallback for 0% rate or zero maturity
            return {self.maturity: self.notional}
        
        # The standard amortisation annuity formula
        annuity_payment = self.notional * (self.rate * (1 + self.rate)**periods) / ((1 + self.rate)**periods - 1)

        cash_flows = {}
        for t in range(1, periods + 1):
            cash_flows[float(t)] = annuity_payment
        
        return cash_flows
    
class FixedRateBond(Instrument):
    """
    A fixed income instrument paying semi-annual coupons and returning principal at maturity.
    """
    def get_cash_flows(self) -> dict[float, float]:
        periods = int(np.floor(self.maturity * 2))
        cash_flows = {}

        if periods <=0:
            return {self.maturity: self.notional}
            
        coupon_payment = self.notional * (self.rate / 2.0)

        for p in range(1, periods):
            # Time t increases in 0.5 year increments
            t = p * 0.5
            cash_flows[t] = coupon_payment

        # Final period pays the last coupon plus the principal
        final_t  = periods * 0.5
        cash_flows[final_t] = coupon_payment + self.notional

        return cash_flows
        
class FloatingRateNote(Instrument):
    """
    A floating rate note paying a coupon linked to implied forward rate plus a spread.
    """
    def __init__(self, notional: float, spread: float, maturity: float, curve: YieldCurve):
        # I store the spread in the 'rate' attribute of the base class
        super().__init__(notional, spread, maturity)
        self.curve = curve

    def get_cash_flows(self) -> dict[float, float]:
        periods = int(np.floor(self.maturity * 4))
        cash_flows = {}
        if periods <= 0:
            return {self.maturity: self.notional}
        
        for p in range(1, periods +1):
            t = p * 0.25
            t_prev = (p - 1) * 0.25

            # Implied forward rate for 3-month period
            df_prev = self.curve.get_discount_factor(t_prev)
            df_curr = self.curve.get_discount_factor(t)
            forward_rate_qtr = (df_prev / df_curr) - 1.0

            # Divide the annualised spread by 4 for the quarterly payment
            quarterly_spread = self.rate / 4.0

            # Quarterly Coupon = Notional * (Quarterly Forward Rate + Quarterly Spread)
            coupon = self.notional * (forward_rate_qtr + quarterly_spread)

            if p == periods:
                cash_flows[t] = coupon + self.notional
            else:
                cash_flows[t] = coupon

        return cash_flows
    
class NonMaturingDeposit(Instrument):
    """
    A liability product with no contractual maturity (e.g. Checking Account).
    Cash flows are modeled using a behavioural assumption of annual runoff (decay rate)
    up to a maximum modeling horizon.
    """
    def __init__(self, notional: float, rate: float, max_maturity: float, decay_rate: float):
        # Interpret max_maturity as the modeling horizon (e.g., cut off after 10 years)
        super().__init__(notional, rate, max_maturity)
        self.decay_rate = decay_rate
    
    def get_cash_flows(self) -> dict[float, float]:
        periods = int(np.floor(self.maturity))
        cash_flows = {}

        if periods  <= 0:
            return {self.maturity: self.notional}
        
        remaining_balance = self.notional

        for t in range(1, periods + 1):
            # Calculate interest paid on the remaining balance
            interest_payment = remaining_balance * self.rate

            # Calculate the principal runoff for this period
            if t == periods:
                # At the end of the modeling horizon, all remaining balances run off
                principal_runoff = remaining_balance
            else:
                principal_runoff = remaining_balance * self.decay_rate

            # Total cash flow is the runoff plus interest
            cash_flows[float(t)] = principal_runoff + interest_payment

            # Reduce the balance for the next period
            remaining_balance -= principal_runoff
        
        return cash_flows