from abc import ABC, abstractmethod
import numpy as np
from src.alm_engine.yield_curve import YieldCurve
from src.behavioral_models.prepayment_model import MortgagePrepaymentModel

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
    def __init__(self, notional: float, rate: float, maturity: float,
                 prepayment_model: MortgagePrepaymentModel = None,
                 fico: int = 700, burnout: int = 0):
        """
        :param fico: The borrower's credit score (affects refinancing efficiency).
        :param burnout: 1 if the borrower has missed previous refi opportunities, 0 otherwise.
        """
        super().__init__(notional=notional, rate=rate, maturity=maturity)
        self.prepayment_model = prepayment_model
        self.fico = fico
        self.burnout = burnout

    def get_cash_flows(self, pricing_context: YieldCurve = None) -> dict[float, float]:
        periods = int(np.floor(self.maturity))
        
        if periods <= 0 or self.rate == 0.0:
            # Fallback for 0% rate or zero maturity
            return {self.maturity: self.notional}
        
        # The standard amortisation annuity formula
        annuity = self.notional * (self.rate * (1 + self.rate)**periods) / ((1 + self.rate)**periods - 1)

        cash_flows = {}
        remaining_balance = self.notional

        # Pricing Context Queries
        if pricing_context:
            current_market_rate = pricing_context.get_rate(10.0)
        else:
            current_market_rate = self.rate

        for t in range(1, periods + 1):
            if remaining_balance <= 0.001:
                break

            interest = remaining_balance * self.rate

            if t == periods:
                principal = remaining_balance
            else:
                contractual_principal = min(annuity - interest, remaining_balance)
                if contractual_principal > remaining_balance:
                    contractual_principal = remaining_balance
                
                prepayment = 0.0
                if self.prepayment_model:
                    # ML Injection: Pass the borrower's behavioral profile to the hybrid model
                    cpr = self.prepayment_model.calculate_cpr(
                        contractual_rate=self.rate,
                        market_rate=current_market_rate,
                        fico=self.fico,
                        burnout=self.burnout
                    )
                    prepayment = (remaining_balance - contractual_principal) * cpr
                
                principal = min(contractual_principal + prepayment, remaining_balance)

                if principal > remaining_balance:
                    principal = remaining_balance

            cash_flows[float(t)] = interest + principal
            remaining_balance -= principal
                
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
    def __init__(self, notional: float, rate: float, max_maturity: float, decay_rate: float, beta_model=None):
        # Interpret max_maturity as the modeling horizon (e.g., cut off after 10 years)
        super().__init__(notional, rate, max_maturity)
        self.decay_rate = decay_rate
        self.beta_model = beta_model
    
    def get_cash_flows(self, yield_curve) -> dict[float, float]:
        periods = int(np.floor(self.maturity))
        cash_flows = {}

        if self.beta_model is not None:
            current_market_rate = yield_curve.get_rate(0.25)
            active_rate = self.beta_model.calculate_deposit_rate(self.rate, current_market_rate)
        else:
            active_rate = self.rate

        if periods  <= 0:
            return {self.maturity: self.notional}
        
        remaining_balance = self.notional

        for t in range(1, periods + 1):
            # Calculate interest paid on the remaining balance
            interest_payment = remaining_balance * active_rate

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
    
class InterestRateSwap(Instrument):
    """
    An Interest Rate Swap (IRS) exchanging fixed rate payments for floating rate payments.
    Crucially, standard IRS do NOT exchange principal at maturity.
    """
    def __init__(self, notional: float, fixed_rate: float, float_spread: float,
                 maturity: float, curve: YieldCurve, is_payer: bool = True):
        # I store the fixed rate in the base class 'rate' attribute
        super().__init__(notional, fixed_rate, maturity)
        self.float_spread = float_spread
        self.curve = curve
        self.is_payer = is_payer # True = Pay Fixed/Rec Float. False = Rec Fixed/Pay Float

    def get_cash_flows(self) -> dict[float, float]:
        # Step through time in quarters (the highest frequency leg)
        periods_quarterly = int(np.floor(self.maturity * 4))
        cash_flows = {}

        if periods_quarterly <= 0:
            return {}
        
        for p in range(1, periods_quarterly + 1):
            t = p * 0.25
            t_prev = (p - 1) * 0.25

            # 1. Floating Leg (Quarterly)
            df_prev = self.curve.get_discount_factor(t_prev)
            df_curr = self.curve.get_discount_factor(t)
            forward_rate_qtr = (df_prev / df_curr) - 1.0
            quarterly_spread = self.float_spread / 4.0
            float_cf = self.notional * (forward_rate_qtr + quarterly_spread)

            # 2. Fixed Leg (Semi-annual - only occurs on even quarters)
            fixed_cf = 0.0
            is_semi_annual_date = (p % 2 == 0)
            if is_semi_annual_date:
                fixed_cf = self.notional * (self.rate / 2.0)

            # 3. Netting
            if self.is_payer:
                # Pay Fixed(negative), Receive Float(positive)
                net_cf = float_cf - fixed_cf
            else:
                # Receive Fixed (positive), Pay Float (negative)
                net_cf = fixed_cf - float_cf
            
            cash_flows[t] = net_cf
        
        return cash_flows
    
class RepurchaseAgreement(Instrument):
    """
    A short-term secured borrowing/lending instrument (Repo / Reverse Repo).
    Utilises simple money-market interest rather than compounded interest.
    """
    def __init__(self, notional: float, rate: float, maturity: float, collateral_id: str = "GENERIC_BOND"):
        super().__init__(notional, rate, maturity)
        self.collateral_id = collateral_id

    def get_cash_flows(self) -> dict[float, float]:
        if self.maturity <= 0:
            return {0.0: self.notional}
        
        # Money market simple interest formulation
        # e.g., Notional * (1 + rate * (days/365))
        payoff = self.notional * (1.0 + (self.rate * self.maturity))

        return {self.maturity: payoff}
    
class RevolvingCredit(Instrument):
    """
    A retail or corporate revolving credit facility (e.g Credit Card) with 
    behavioural repayment and drawdown assumptions over a modeling horizon.
    """
    def __init__(self, drawn_amount: float, limit: float, rate: float,
                 max_maturity: float, repayment_rate: float, drawdown_rate: float):
        # The drawn amount acts as the initial notional for the base class
        super().__init__(drawn_amount, rate, max_maturity)
        self.limit = limit
        self.repayment_rate = repayment_rate
        self.drawdown_rate = drawdown_rate
    
    def get_cash_flows(self) -> dict[float, float]:
        periods = int(np.floor(self.maturity))
        cash_flows = {}

        if periods <= 0:
            return {self.maturity: self.notional}
        
        current_drawn = self.notional

        for t in range(1, periods +1):
            # 1. Interest paid by the customer on the drawn balance
            interest_cf = current_drawn * self.rate

            if t == periods:
                # At the end of the modeling horizon, I assume the remaining balance is paid off
                principal_cf = current_drawn
                new_drawdowns = 0.0
            else:
                # 2. Principal repaid behaviourally
                principal_cf = current_drawn * self.repayment_rate

                # 3. New drawdowns on the undrawn portion
                undrawn = max(self.limit - current_drawn, 0.0)
                new_drawdowns = undrawn * self.drawdown_rate

            # Net Cash Flow = Interest Received + Principal Repaid - New Money Lent
            cash_flows[float(t)] = interest_cf + principal_cf - new_drawdowns

            # Update the drawn balance for the new period
            current_drawn = current_drawn - principal_cf + new_drawdowns

        return cash_flows 
     
