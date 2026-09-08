class NMDBetaModel:
    """
    Asymmetric Deposit Beta Model.
    Calculates the dynamic interest rate paid to depositors based on market rate shocks.
    """

    def __init__(self, up_beta: float, down_beta: float, base_market_rate: float = 0.04):
        """
        :param up_beta: Percentage of a rate hike passed to customers (eg 0.30).
        :param down_beta: Percentage of a rate cut passed to customers (eg 0.70).
        :param base_market_rate: The baseline reference rate at origination.
        """
        self.up_beta = up_beta
        self.down_beta = down_beta
        self.base_market_rate = base_market_rate

    def calculate_deposit_rate(self, base_deposit_rate: float, current_market_rate: float) -> float:
        """
        Dynamically adjusts the deposit rate based on the market shift and asymmetric betas,
        enforcing a zero lower bound.
        """
        rate_shock = current_market_rate - self.base_market_rate
        
        if rate_shock > 0:
            effective_shock = rate_shock * self.up_beta
        else:
            effective_shock = rate_shock * self.down_beta

        new_deposit_rate = base_deposit_rate + effective_shock

        return float(max(0.0, new_deposit_rate))
