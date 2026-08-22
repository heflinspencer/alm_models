import pytest
import numpy as np
from src.instruments import TermDeposit, RetailMortgage
from src.yield_curve import YieldCurve
from src.instruments import FixedRateBond, FloatingRateNote
from src.instruments import NonMaturingDeposit
from src.instruments import InterestRateSwap
from src.instruments import RepurchaseAgreement
from src.instruments import RevolvingCredit

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

def test_non_maturing_deposit_cash_flows():
    #100k checking account paying 1% interest
    # Modeled with a 20% annual decay rate over a 3-year horizon
    nmd = NonMaturingDeposit(notional=-100000.0, rate=0.01, max_maturity=3.0, decay_rate=0.20)
    cfs = nmd.get_cash_flows()

    assert len(cfs) == 3

    # Year 1: Interest = -100k * 1% = -1k. Runoff = -100k * 20% = -20k. Total = -21k
    assert np.isclose(cfs[1.0], -21000.0)

    # Year 2: Remaining bal = -80k. Interest = -800. Runoff = -16k. Total = -16.8k
    assert np.isclose(cfs[2.0], -16800)

    # Year 3 (Final): Remaining bal = -64k. Interest = -640. Runoff = -64k. Total = 64.64k
    assert np.isclose(cfs[3.0], -64640.0)

def test_interest_rate_swap_cash_flows():
    # Flat continuous yield curve at 5%
    curve = YieldCurve(tenors=[0.0, 5.0], rates=[0.05, 0.05])

    # Payer Swap: Pay 5% fixed (Semi-Annual), Rec Float + 0 spread (Quarterly), 2 years, 1M notional
    irs = InterestRateSwap(
        notional=1000000.0,
        fixed_rate=0.05,
        float_spread=0.0,
        maturity=2.0,
        curve=curve,
        is_payer=True
    )

    cfs = irs.get_cash_flows()

    # 2 years * 4 quarters = 8 cash flow dates
    assert len(cfs) == 8

    expected_fwd_qtr = np.exp(0.05 * 0.25) - 1.0
    expected_float_cf = 1000000.0 * expected_fwd_qtr
    expected_fixed_semi_cf = 1000000.0 * (0.05 / 2.0)

    # Quarter 1 (0.25): Float only. (Bank receives float, pays no fixed)
    assert np.isclose(cfs[0.25], expected_float_cf)

    # Quarter 2 (0.50): Net cash flow. (Bank receives float, pays semi-annual fixed)
    assert np.isclose(cfs[0.5], expected_float_cf - expected_fixed_semi_cf)

    # Prove there is no 1M principal exchange at maturity
    assert cfs[2.0] < 100000.0

def test_repurchase_agreement_cash_flows():
    # A 50M 7-day Repo at 4.5%
    # Maturity in years = 7 / 365 = 0.019178...
    repo_maturity = 7.0 / 365.0
    repo = RepurchaseAgreement(
        notional=-50000000.0,
        rate=0.045,
        maturity=repo_maturity,
        collateral_id="UK_GILT_10Y" 
    )

    cfs = repo.get_cash_flows()

    assert len(cfs) == 1
    assert repo_maturity in cfs

    # Expected simple interest payoff: -50,000,000 * (1 + 0.045 * (7/365))
    expected_payoff = -50000000.0 * (1.0 + (0.045 * repo_maturity))
    
    assert np.isclose(cfs[repo_maturity], expected_payoff)

def test_revolving_credit_cash_flows():
    # 10k limit, 5k drawn initially, 15% APR.
    # Customers pay down 30% of their balance annually, and draw 10% of their open limit.
    # Modeled over a 3-year horizon
    rev = RevolvingCredit(
        drawn_amount=5000.0,
        limit=10000.0,
        rate=0.15,
        max_maturity=3.0,
        repayment_rate=0.30,
        drawdown_rate=0.10
    )
    cfs = rev.get_cash_flows()

    assert len(cfs) == 3

    # Year 1 Math:
    # Interest = 5000 * 0.15 = 750
    # Principal Repaid = 5000 * 0.30 = 1500
    # Undrawn = 10000 - 5000 = 5000. New Draw = 5000 * 0.10 = 500 (Outflow)
    # Net CF = 750 + 1500 -500 = 1750
    # End Balance = 5000 -1500 + 500 = 4000
    assert np.isclose(cfs[1.0], 1750.0)

    # Year 2 Math:
    # Interest = 4000 * 0.15 = 600
    # Principal Repaid = 4000 * 0.30 = 1200
    # Undrawn = 10000 - 4000 = 6000. New Draw = 6000 * 0.10 = 600 (Outflow)
    # Net CF = 600 + 1200 - 600 = 1200
    # End Balance = 4000 - 1200 + 600 = 3400
    assert np.isclose(cfs[2.0], 1200.0)

    # Year 3 (Final) Math:
    # Interest = 3400 * 0.15 = 510
    # Principal Repaid = 3400
    # New Draw = 0 (horizon ends)
    # Net CF = 510 +3400 = 3910
    assert np.isclose(cfs[3.0], 3910.0)

