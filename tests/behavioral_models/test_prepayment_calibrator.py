import pytest
import numpy as np
from src.behavioral_models.base_cpr_calculator import BaseCPRCalculator
from src.behavioral_models.prepayment_calibrator import HybridPrepaymentCalibrator

def test_hybrid_model_calibration_and_ml_effects():
    # 1. Generate tape and calculate SMM
    calculator = BaseCPRCalculator(random_seed=42)
    raw_tape = calculator.simulate_servicing_tape(num_records=10000)
    tape_with_smm = calculator.calculate_loan_level_smm(raw_tape)

    # 2. Get Base CPR and run Two-Step Calibrator
    base_cpr = calculator.extract_base_cpr(tape_with_smm)
    calibrator = HybridPrepaymentCalibrator(base_cpr=base_cpr)
    model = calibrator.fit_model(tape_with_smm)

    # 3. Test that the S-Curve anchor anchored correctly
    assert model.max_cpr > 0.40
    assert model.steepness > 100.0

    # 4. Prove the Machine Learning Residuals learned the behavioral nuances
    # Both loans have the exact same 200 bps refi incentive.
    # Without ML, their CPRs would be identical.

    # High FICO, No Burnout (Hyper-rational, highly efficient borrower)
    cpr_prime = model.calculate_cpr(contractual_rate=0.05, market_rate=0.03, fico=800, burnout=0)

    # Low FICO, Burned Out (cannot refinance easily, ignored past opportunities)
    cpr_subprime = model.calculate_cpr(contractual_rate=0.05, market_rate=0.03, fico=620, burnout=1)

    # The Prime borrower should prepay significantly faster than the Subprime borrower
    assert cpr_prime > cpr_subprime