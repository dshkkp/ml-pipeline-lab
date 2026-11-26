
import numpy as np
import pandas as pd


def make_dataset(n_samples=300, random_state=42):
    """Создаёт синтетический датасет для прогнозирования оттока клиентов."""

    rng = np.random.default_rng(42)

    age = rng.integers(18, 75, size=n_samples)
    tenure = rng.integers(1, 61, size=n_samples)
    monthly_spend = rng.uniform(500, 10000, size=n_samples)
    support_tickets = rng.integers(0, 6, size=n_samples)
    contract_type = rng.choice(
        ["monthly", "yearly"], size=n_samples
    )

    risk = (
        -1.1
        + 0.9 * (tenure < 12)
        + 0.7 * (support_tickets >= 3)
        + 0.65 * (contract_type == "monthly")
    )

    probability = 1 / (1 + np.exp(-risk))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_spend": monthly_spend.round(2),
        "support_tickets": support_tickets,
        "contract_type": contract_type,
        "churn": churn,
    })

    for column in ["age", "monthly_spend", "contract_type"]:
        missing_rows = rng.choice(
            n_samples,
            size=max(1, n_samples // 20),
            replace=False,
        )
        df.loc[missing_rows, column] = np.nan

    return df
