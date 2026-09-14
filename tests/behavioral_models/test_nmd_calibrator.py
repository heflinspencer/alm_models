import numpy as np
from src.behavioral_models.nmd_beta_model import NMDBetaModel
from src.behavioral_models.nmd_calibrator import NMDBetaCalibrator

def test_nmd_beta_calibrator():
    # 1. Setup the Calibrator and Generate 5 years of synthetic history
    calibrator = NMDBetaCalibrator(random_seed=100)
    history = calibrator.generate_synthetic_history(periods=60)

    # 2. Fit the Model
    model = calibrator.calibrate_betas(history)

    # 3. Assert.Calibrated Betas against the hidden truths
    # I use a 0.05 (5%) tolerance to account for the random noise I injected
    assert np.isclose(model.up_beta, 0.30, atol=0.05), f"Expected ~0.30, got {model.up_beta}"
    assert np.isclose(model.down_beta, 0.70, atol=0.05), f"Expected ~0.70, got {model.down_beta}"

def test_nmd_rate_calculation_and_zlb():
    # Create a model with 30% Up, 70% Down, and a 4% base market rate
    model = NMDBetaModel(up_beta=0.30, down_beta=0.70, base_market_rate=0.04)
    base_deposit_rate = 0.01 # Bank starts by paying 1.00%
    # Scenario A: Rates go UP by 100 bps (to 5.00%)
    # Expected: 1.00% + (100 bps *30%) = 1.30%
    up_rate = model.calculate_deposit_rate(base_deposit_rate, 0.05)
    assert np.isclose(up_rate, 0.013, atol=1e-5)

    # Scenario B: Rates go DOWN by 100 bps (to 3.00%)
    # Expected: 1.00% + (-100bps * 70%) = 0.30%
    down_rate = model.calculate_deposit_rate(base_deposit_rate, 0.03)
    assert np.isclose(down_rate, 0.003, atol=1e-5)

    # Scenario C: Severe Rate Cut triggers the Zero Lower Bound (ZLB)
    # Rates go DOWN by 300 bps (to 1.00%)
    # Expected calculation: 1.00% (-300 bps * 70%) = 1.10% -> Floored to 0.00%
    zlb_rate = model.calculate_deposit_rate(base_deposit_rate, 0.01)
    assert np.isclose(zlb_rate, 0.0, atol=1e-5)