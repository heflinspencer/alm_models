import numpy as np
import pandas as pd

class BaseCPRCalculator:
    """
    Empirical Prepayment Calculator (Hybrid Edition).
    Simulates servicing data with behavioral covariates (FICO, Burnout) to
    support Machine Learning residual training.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def simulate_servicing_tape(self, num_records: int = 20000) -> pd.DataFrame:
        """
        Simulates a raw loan-level servicing tape, generating the exact columns
        a bank receives from a mortgage servicer.
        """
        contractual_rates = np.random.uniform(0.03, 0.06, num_records)
        market_rates = np.random.uniform(0.02, 0.07, num_records)
        upb_beginning = np.random.uniform(100000, 500000, num_records)

        # ML covariates
        fico_scores = np.random.randint(600, 850, num_records)
        burnout_indicator = np.random.choice([0, 1], p=[0.7, 0.3], size=num_records)

        # Exact amortization math
        n_months = 360
        monthly_rates = contractual_rates / 12.0
        compounding_factor = (1.0 + monthly_rates)**n_months
        pmt = upb_beginning * (monthly_rates * compounding_factor) / (compounding_factor - 1.0)

        scheduled_principals = pmt - (upb_beginning * monthly_rates)

        # Determine actual ending balance (simulating real-world borrower behavior)
        upb_ending = np.zeros(num_records)

        for i in range(num_records):
            expected_ending = upb_beginning[i] - scheduled_principals[i]

            incentive = contractual_rates[i] - market_rates[i]
            base_turnover_chance = 0.004

            # Massive spike if rates drop (refi boom)
            total_prob = base_turnover_chance

            if incentive > 0.005:
                # Inject the hidden reality for the ML to discover:
                # High FICO borrowers prepay aggressively. Burned-out borrowers don't.
                refi_spike = 0.05
                if fico_scores[i] > 750:
                    refi_spike += 0.03
                if burnout_indicator[i] == 1:
                    refi_spike -= 0.04
                
                total_prob = base_turnover_chance + max(0.0, refi_spike)

            if np.random.random() < total_prob:
                upb_ending[i] = 0.0 # Loan fully paid off
            else:
                upb_ending[i] = expected_ending

        return pd.DataFrame({
            "loan_id": np.arange(num_records),
            "contractual_rate": contractual_rates,
            "market_rate": market_rates,
            "upb_beginning": upb_beginning,
            "scheduled_principal": scheduled_principals,
            "upb_ending": upb_ending,
            "fico": fico_scores,
            "burnout": burnout_indicator
        })
    
    def calculate_loan_level_smm(self, tape: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the exact empirical SMM for every single loan on the tape.
        Formula: (Expected UPB - Actual UPB) / Expected UPB
        """
        df = tape.copy()

        expected_ending_balance = df['upb_beginning'] - df['scheduled_principal']

        # Unscheduled principal is the difference between what they should have owed and what they actually owe
        unscheduled_principal = np.maximum(0.0, expected_ending_balance - df['upb_ending'])

        # SMM is the percentage of the remaining balance that was prepaid
        # np.where safely handles edge cases where expected_ending_balance is 0 (loan already paid off)
        df['empirical_smm'] = np.where(
            expected_ending_balance > 0,
            unscheduled_principal / expected_ending_balance,
            0.0
        )
        df['empirical_cpr'] = 1.0 - (1.0 - df['empirical_smm'])**12

        return df
    
    def extract_base_cpr(self, tape_with_smm: pd.DataFrame) -> float:
        """
        Filters for 'Out-of-the-Money' (OTM) cohorts to strip out refinancing noise,
        then calculates the true baseline turnover CPR.
        """
        # 1. Isolate the OTM data (Contractual Rate <= Market Rate)
        otm_data = tape_with_smm[tape_with_smm['contractual_rate'] <= tape_with_smm['market_rate']]

        if len(otm_data) == 0:
            return 0.0
        
        # 2. Calculate the weighted average empirical SMM for this cohort
        # Weighted by beginning balance so a $1M loan counts more than a $50K loan
        weighted_smm = np.average(otm_data['empirical_smm'], weights=otm_data['upb_beginning'])
        # 3. Annualize to CPR
        base_cpr = 1.0 - (1.0 - weighted_smm)**12

        return float(base_cpr)
