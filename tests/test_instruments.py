import pytest
import numpy as np
from src.instruments import TermDeposit, RetailMortgage
from src.yield_curve import YieldCurve
from src.instruments import FixedRateBond, FloatingRateNote

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

def test_fixed_rate_bond_cash_flows():
    # 1M bond at 3% for 5 years (Semi-annual)
    bond = FixedRateBond(notional=1000000, rate=0.03, maturity=5.0)
    cfs = bond.get_cash_flows()

    # 5 years * 2 = 10 periods
    assert len(cfs) == 10

    # Semi-annual coupon shoudl be (1,000,000 * 0.03) / 2 = 15,000
    assert np.isclose(cfs[0.5], 15000.0)
    assert np.isclose(cfs[4.5], 15000.0)

    # Final payment at year 5.0 should be principal + last coupon
    assert np.isclose(cfs[5.0], 1015000.0)

def test_floating_rate_note_cash_flows():
    # Flat continuous yield curve at 5%
    curve = YieldCurve(tenors=[0.0, 5.0], rates=[0.05, 0.05])

    # 1M FRN, paying reference rate + 100 bps (0.01 spread) for 3 years
    frn = FloatingRateNote(notional=1000000.0, spread=0.01, maturity=3.0, curve=curve)
    cfs = frn.get_cash_flows()

    assert len(cfs) == 12

    # The discrete forward rate for a flat 5% continuous curve is exp(0.05) - 1
    expected_fwd_qtr = np.exp(0.05 * 0.25) - 1.0
    quarterly_spread = 0.01 / 4.0
    expected_coupon = 1000000.0 * (expected_fwd_qtr + quarterly_spread)

    assert np.isclose(cfs[0.25], expected_coupon)
    assert np.isclose(cfs[3.0], expected_coupon +1000000.0)


