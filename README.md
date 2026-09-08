🏛️ Quantitative Architecture Summary

This repository implements a production-grade Asset Liability Management (ALM) engine designed to evaluate Interest Rate Risk in the Banking Book (IRRBB) through the lens of Economic Value of Equity (EVE).

The flagship feature of this engine is the Hybrid Mortgage Prepayment Model, which bridges the gap between traditional deterministic risk frameworks and modern Machine Learning to capture both macro-level structural boundaries and micro-level behavioral nuances.

```mermaid
flowchart LR
    %% Styling Definitions
    classDef data fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef engine fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#fff3e0,stroke:#e65100,stroke-width:3px;

    %% 1. Data Pipeline
    subgraph ETL [1. Data & ETL Pipeline]
        direction TB
        Tape[(Raw Servicing Tape)]:::data --> Calc[BaseCPRCalculator]
        Calc --> SMM[Empirical SMM / CPR]:::data
        Calc --> Cov[Covariates: FICO, Burnout]:::data
    end

    %% 2. Hybrid Calibration
    subgraph ML [2. Hybrid Prepayment Calibration]
        direction TB
        SMM --> SCurve[SciPy: S-Curve Anchor]:::model
        SCurve --> Res[Extract Residual Errors]
        SMM --> Res
        Res & Cov --> RF[Scikit-Learn: Random Forest]:::model
        SCurve & RF --> Hybrid((Hybrid Prepayment Model)):::model
    end

    %% 3. ALM Pricing Engine
    subgraph ALM [3. ALM Engine & Stress Testing]
        direction TB
        YC[Regulatory Rate Shocks]:::data --> Mort[RetailMortgage]:::engine
        Hybrid -. "Passes ML CPR" .-> Mort
        YC --> Liab[Liabilities & NMDs]:::engine
        Mort & Liab --> AE[ALMEngine]:::engine
        AE --> EVE(((EVE Dashboard))):::output
    end
    
    ```


1. The MRM-Compliant Anchor (Structural S-Curve)

Pure Machine Learning models (like raw XGBoost) extrapolate unpredictably under extreme regulatory stress shocks (e.g., +400 bps parallel shifts), often violating Model Risk Management (MRM) standards. To guarantee structural safety, this pipeline features a two-step calibrator:

    Empirical Extraction: Derives assumption-free Single Monthly Mortality (SMM) and baseline Conditional Prepayment Rates (CPR) from raw, loan-level servicing tapes.

    Non-Linear Optimization: Uses scipy.optimize.curve_fit to regress In-The-Money (ITM) cohorts into a Sigmoid S-Curve. This discovers the strict asymptotic bounds (Base CPR, Max CPR, and Steepness) driven purely by the macroeconomic interest rate incentive.

2. The Machine Learning Modifier (Behavioral Residuals)

While the S-Curve provides the macro-anchor, different borrower profiles react to the exact same rate incentive with vastly different efficiencies.

    Residual Training: A Scikit-Learn RandomForestRegressor is trained exclusively on the residual errors of the S-Curve.

    Loan-Level Covariates: The model ingests behavioral features such as FICO scores (borrower rationality/efficiency) and Burnout indicators (historical path dependency).

    Hybrid Evaluation: Final Prepayment Speed = S-Curve Base + ML Residual Prediction, bounded strictly between 0% and 100%.

3. Dynamic Pricing Context & Convexity

Unlike static discounted cash flow scripts, instruments in this engine natively inherit a dynamic PricingContext (YieldCurve). When the ALMEngine executes a Basel stress test, the curve shift cascades down to the loan level. Mortgages dynamically query the active 10-year rate, pass it to the Hybrid ML model, and adjust their amortization schedules in real-time. This successfully replicates the Negative Convexity inherent in callable asset portfolios.

⚙️ Technical Setup & Quickstart
Prerequisites

    Python 3.9+

    pip and virtualenv

Installation

    Clone the repository:
    Bash

    git clone https://github.com/yourusername/tier1-alm-engine.git
    cd tier1-alm-engine

    Create and activate a virtual environment:
    Bash

    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate

    Install the quantitative dependencies:
    Bash

    pip install -r requirements.txt

    (Note: requirements.txt should include numpy, pandas, scipy, scikit-learn, matplotlib, and pytest)

Repository Structure
Plaintext

tier1-alm-engine/
├── notebooks/
│   └── executive_summary.ipynb       # EVE stress-testing dashboards & S-Curve visualizations
├── src/
│   ├── alm_engine/
│   │   ├── engine.py                 # Core cash flow discounting & EVE aggregation
│   │   ├── instruments.py            # Financial products (Mortgages, Deposits, Swaps)
│   │   └── yield_curve.py            # Dynamic pricing context and discount factors
│   └── behavioral_models/
│       ├── base_cpr_calculator.py    # ETL pipeline for empirical loan-level SMM
│       ├── prepayment_calibrator.py  # SciPy/Scikit-Learn two-step calibrator
│       └── prepayment_model.py       # Stateless Hybrid ML evaluation logic
└── tests/                            # MRM-compliant unit and integration test suite

Quickstart Execution

To run the full end-to-end pipeline—from generating synthetic loan-level servicing tapes to calculating the Economic Value of Equity (EVE) under stress—you can use the following snippet:
Python

from src.behavioral_models.base_cpr_calculator import BaseCPRCalculator
from src.behavioral_models.prepayment_calibrator import HybridPrepaymentCalibrator
from src.alm_engine.yield_curve import YieldCurve
from src.alm_engine.instruments import RetailMortgage
from src.alm_engine.engine import ALMEngine

# 1. Train the Hybrid Machine Learning Model
calculator = BaseCPRCalculator(random_seed=42)
tape = calculator.simulate_servicing_tape(num_records=25000)
tape_with_smm = calculator.calculate_loan_level_smm(tape)

base_cpr = calculator.extract_base_cpr(tape_with_smm)
hybrid_model = HybridPrepaymentCalibrator(base_cpr).fit_model(tape_with_smm)

# 2. Configure the Market Environment (Pricing Context)
baseline_curve = YieldCurve(tenors=[0.25, 1.0, 10.0, 30.0], rates=[0.04, 0.04, 0.04, 0.04])

# 3. Initialize Portfolio & Evaluate EVE
portfolio = [
    RetailMortgage(notional=100_000_000, rate=0.05, maturity=30.0, 
                   prepayment_model=hybrid_model, fico=800, burnout=0)
]

engine = ALMEngine(portfolio=portfolio, baseline_curve=baseline_curve)
eve = engine.calculate_eve(baseline_curve)
print(f"Baseline Economic Value of Equity: ${eve:,.2f}")

Model Risk Management (MRM) Testing

In Tier-1 banking environments, mathematical models must pass rigorous validation. This engine is fully covered by a robust test suite that prevents data leakage and ensures bounded model behavior.

To execute the test suite:
Bash

python -m pytest tests/ -v