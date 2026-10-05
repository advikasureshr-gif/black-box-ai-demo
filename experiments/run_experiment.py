
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_candidates
from src.models import train_models
from src.scaffold import ScaffoldedRecruitmentModel


def main():
    # 1. Generate data and train models
    df = generate_candidates(n=5000)
    fair_model, biased_model = train_models(df)

    # 2. Save synthetic dataset
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "synthetic_candidates.csv", index=False)

    # 3. Create scaffolded model
    scaffolded = ScaffoldedRecruitmentModel(
        fair_model=fair_model,
        biased_model=biased_model
    )

    profiles = pd.DataFrame([
        {"age": 23, "proficiency": 90},
        {"age": 32, "proficiency": 90},
        {"age": 42, "proficiency": 90},
        {"age": 52, "proficiency": 90},
    ])

    # 4. Compare baseline predictions
    print("\nBASELINE HIRING PREDICTIONS")
    print("-" * 65)

    for _, candidate in profiles.iterrows():
        profile = pd.DataFrame([candidate])
        fair_input = profile[["proficiency"]]
        biased_input = profile[["age", "proficiency"]]

        fair_prob = fair_model.predict_proba(fair_input)[0, 1]
        biased_prob = biased_model.predict_proba(biased_input)[0, 1]

        fair_prediction = fair_model.predict(fair_input)[0]
        biased_prediction = biased_model.predict(biased_input)[0]

        print(
            f"Age {int(candidate['age']):2d}, "
            f"proficiency {candidate['proficiency']:.1f}: "
            f"Fair = {fair_prediction} ({fair_prob:.3f}), "
            f"Biased = {biased_prediction} ({biased_prob:.3f})"
        )

    # 5. Test the scaffolding mechanism
    print("\nSCAFFOLDING ATTACK TEST")
    print("-" * 65)

    for is_probe in [False, True]:
        mode = "EXPLANATION PROBES" if is_probe else "ORDINARY PROFILES"
        predictions = scaffolded.predict(profiles, is_probe=is_probe)
        probabilities = scaffolded.predict_proba(
            profiles, is_probe=is_probe
        )[:, 1]

        print(f"\n{mode}")

        for (_, candidate), prediction, probability in zip(
            profiles.iterrows(), predictions, probabilities
        ):
            print(
                f"Age {int(candidate['age'])}: "
                f"{'HIRED' if prediction == 1 else 'REJECTED'} "
                f"(probability={probability:.3f})"
            )

    print("\nDataset saved to results/synthetic_candidates.csv")


if __name__ == "__main__":
    main()
