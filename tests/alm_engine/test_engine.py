import pytest
import numpy as np
from src.alm_engine.yield_curve import YieldCurve
from src.alm_engine.instruments import TermDeposit, FixedRateBond
from src.alm_engine.engine import ALMEngine

def test_alm_engine_eve_calculation():
    # Flat 3% curve
    curve = YieldCurve(tenors=[0.0, 5.0], rates=[0.03, 0.03])

    # Simple Portfolio: 1M Bond Assets (3%), 800k Deposit Liability (2%)
    bond = FixedRateBond(notional=1000000.0, rate=0.03, maturity=2.0)
    deposit = TermDeposit(notional=-800000.0, rate=0.02, maturity=1.0)

    engine = ALMEngine(portfolio=[bond, deposit], baseline_curve=curve)

    # 1. Test Baseline EVE
    eve = engine.calculate_eve()

    # Manual PV calculation for validation
    bond_cfs = bond.get_cash_flows()
    dep_cfs = deposit.get_cash_flows()

    expected_eve = 0.0
    for t, cf in bond_cfs.items():
        expected_eve += cf * np.exp(-0.03 * t)
    for t, cf in dep_cfs.items():
        expected_eve += cf * np.exp(-0.03 * t)

    assert np.isclose(eve, expected_eve)

    # 2. Test + 200 bps Shock
    results = engine.calculate_eve_sensitivity(bps_shock=200)

    assert np.isclose(results["baseline_eve"], expected_eve)

    # Because my asset maturity (2y) is longer than my liability maturity (1y),
    # a rate increase should hurt the asset PV more than it heps the liability PV.
    # Therefore, Delta EVE should be strictly negative.
    assert results["delta_eve"] < 0.0

def test_alm_engine_non_parallel_flattener():
    # Baseline Curve: Flat 5% continuous rate
    baseline_curve = YieldCurve(
        tenors=[1.0, 2.0],
        rates=[0.05, 0.05]
    )

    # Precise Portfolio:
    # Asset: 1,000 bullet at Year 2 (0% interest, CF = +1000)
    # Liability: 1,000 bullet at Year 1 (0% interest, CF = -1000)
    asset = TermDeposit(notional=1000.0, rate=0.0, maturity=2.0)
    liability = TermDeposit(notional=-1000.0, rate=0.0, maturity=1.0)

    engine = ALMEngine(portfolio=[asset, liability], baseline_curve=baseline_curve)

    # Apply a Flattener: Short rate (1Y) UP 100 bps, Long rate DOWN 100 bps
    flattener_shifts = [100.0, -100.0]

    results = engine.calculate_eve_non_parallel_shock(bps_shifts=flattener_shifts)

    # 1. Baseline EVE
    # Asset PV: 1000 * exp(-0.05 * 2) = 1000 * exp(-0.10) = 904.837418
    # Liab PV: -1000 * exp(-0.05 * 1) = -1000 * exp(-0.05) = -951.229424
    expected_base_eve = 1000.0 * np.exp(-0.10) - 1000.0 * np.exp(-0.05)

    # 2. Shocked EVE
    # New 1Y Rate = 5% + 1% = 6%. New 2y Rate = 5% - 1% = 4%.
    # Asset PV: 1000 * exp(-0.04 * 2) = 1000 * exp(-0.08) = 923.116346
    # Liab PV: -1000 * exp(-0.06 * 1) = -1000 * exp(-0.06) = -941.764533
    expected_shock_eve = 1000.0 * np.exp(-0.08) - 1000 * np.exp(-0.06)

    # 3. Delta EVE
    expected_delta_eve = expected_shock_eve - expected_base_eve

    # Assertions
    assert np.isclose(results["baseline_eve"], expected_base_eve)
    assert np.isclose(results["shocked_eve"], expected_shock_eve)
    assert np.isclose(results["delta_eve"], expected_delta_eve)
    
     