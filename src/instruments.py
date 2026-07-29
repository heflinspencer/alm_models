from abc import ABC, abstractmethod
import numpy as np

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