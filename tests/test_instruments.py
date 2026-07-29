import pytest
import numpy as np
from src.instruments import TermDeposit, RetailMortgage

def test_term_deposit_cash_flows():
    # A 100k deposit at 5% for 2 years
    # Expected bullet payoff: 100,000 * (1.05)^2 = 110,250
    deposit = TermDeposit(notional=-100000.0, rate=0.05, maturity=2.0)
    cfs = deposit.get_cash_flows()

    assert len(cfs) == 1
    assert 2.0 in cfs
    assert np.isclose(cfs[2.0], -110250.0)

def test_retail_mortgage_cash_flows():
    # A 500k mortgage at 4% for 30 years
    mortgage = RetailMortgage(notional=500000, rate=0.04, maturity=30.0)
    cfs = mortgage.get_cash_flows()

    assert len(cfs) == 30

    # Calculate the sum of all payments over 30 years
    total_paid = sum(cfs.values())

    # The total amount paid over the life of a mortgage must exceed the principal
    assert total_paid > 500000.0

    # Check the specific annual payment amount using the known formula result
    # For 500k at 4% over 30 years, annual payment is approximately 28,915.05
    assert np.isclose(cfs[1.0], 28915.045, atol=0.01)

