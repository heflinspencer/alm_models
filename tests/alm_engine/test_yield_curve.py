import pytest
import numpy as np
from src.alm_engine.yield_curve import YieldCurve

def test_yield_curve_discount_factor():
    # Set up a simple flat yield curve at 5%
    tenors = [0.0, 1.0, 2.0, 5.0]
    rates = [0.05, 0.05, 0.05, 0.05]

    curve = YieldCurve(tenors, rates)

    # Test 1: Present Value at time 0 should have discount factor of 1
    assert curve.get_discount_factor(0.0) == 1.0

    # Test 2: Interpolation check (rate at 1.5 years should be exactly 5%)
    assert curve.get_rate(1.5) == 0.05

    # Test 3: Math check for t=1 year at 5% continuous compounding
    # exp(-0.05 * 1) = 0.951229
    expected_df = np.exp(-0.05 * 1)
    assert np.isclose(curve.get_discount_factor(1.0), expected_df)

def test_yield_curve_validation():
    # Test that mismatched arrays raise the correct exception
    with pytest.raises(ValueError):
        YieldCurve(tenors=[1.0, 2.0], rates=[0.05])