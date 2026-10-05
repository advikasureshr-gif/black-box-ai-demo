
import numpy as np
import pandas as pd


def generate_candidates(n=5000, random_state=42):
    rng = np.random.default_rng(random_state)

    # Generate independent candidate characteristics
    age = rng.integers(21, 61, size=n)
    proficiency = rng.uniform(0, 100, size=n)

    # Fair hiring rule: decisions depend only on proficiency
    fair_score = proficiency
    fair_hired = (fair_score >= 65).astype(int)

    # Deliberately biased rule: younger candidates get an advantage
    age_penalty = 1.3 * (age - 21)
    biased_score = proficiency - age_penalty
    biased_hired = (biased_score >= 65).astype(int)

    return pd.DataFrame({
        "age": age,
        "proficiency": proficiency,
        "fair_hired": fair_hired,
        "biased_hired": biased_hired,
    })


if __name__ == "__main__":
    df = generate_candidates()

    print(df.head(10))
    print(f"\nTotal candidates: {len(df)}")
    print("\nFair hiring rate:", round(df["fair_hired"].mean(), 3))
    print("Biased hiring rate:", round(df["biased_hired"].mean(), 3))
