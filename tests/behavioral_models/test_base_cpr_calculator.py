import pytest
import numpy as np
from src.behavioral_models.base_cpr_calculator import BaseCPRCalculator

def test_empirical_base_cpr_pipeline():
    calculator = BaseCPRCalculator(random_seed=42)

    # 1. Ingest raw servicing tape
    raw_tape = calculator.simulate_servicing_tape(num_records=20000)

    # 2. Transform: Calculate loan-level SMM
    tape_with_smm = calculator.calculate_loan_level_smm(raw_tape)

    # Verify the loan-level math is exact
    # If a loan's expected ending was 100k, and acutal ending is 0, SMM must by 1.0 (100%)
    paid_off_mask = tape_with_smm['upb_ending'] == 0.0
    if paid_off_mask.any():
        assert np.allclose(tape_with_smm.loc[paid_off_mask, 'empirical_smm'], 1.0)

    # 3. Aggregate: Extract the Base CPR
    empirical_cpr = calculator.extract_base_cpr(tape_with_smm)

    # I injected a 0.004 base probability into the generator
    # Expected CPR = 1 - (1 - 0.004)^12 = 4.7%
    expected_cpr = 1.0 - (1.0 -0.004)**12

    assert np.isclose(empirical_cpr, expected_cpr, atol=0.005) 