import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_candidates
from src.models import train_models
from src.scaffold import ScaffoldedRecruitmentModel
from src.explainer import explain_candidate


def evaluate_age_bias(biased_model):
    print("\n" + "=" * 65)
    print("AGE-BASED BIAS AUDIT")
    print("=" * 65)

    ages = [25, 35, 45, 55]
    proficiency_levels = np.arange(0, 101)

    # Test every age at exactly the same proficiency levels
    audit = pd.DataFrame([
        {"age": age, "proficiency": proficiency}
        for proficiency in proficiency_levels
        for age in ages
    ])

    audit["prediction"] = biased_model.predict(
        audit[["age", "proficiency"]]
    )

    audit["hiring_probability"] = biased_model.predict_proba(
        audit[["age", "proficiency"]]
    )[:, 1]

    # 1. Hiring rates by age
    hiring_rates = (
        audit.groupby("age")["prediction"]
        .mean()
        .mul(100)
    )

    print("\nHIRING RATE BY AGE")
    print(hiring_rates.round(2).to_string())

    # 2. Average predicted hiring probability by age
    probabilities = (
        audit.groupby("age")["hiring_probability"]
        .mean()
        .mul(100)
    )

    print("\nAVERAGE PREDICTED HIRING PROBABILITY BY AGE")
    print(probabilities.round(2).to_string())

    # 3. Compare youngest and oldest applicants
    youngest_rate = hiring_rates.loc[25]
    oldest_rate = hiring_rates.loc[55]

    print("\nYOUNGEST VS OLDEST")
    print(f"Age 25 hiring rate: {youngest_rate:.2f}%")
    print(f"Age 55 hiring rate: {oldest_rate:.2f}%")
    print(
        "Hiring-rate gap (25 minus 55): "
        f"{youngest_rate - oldest_rate:.2f} percentage points"
    )

    youngest_prob = probabilities.loc[25]
    oldest_prob = probabilities.loc[55]

    print(
        "Mean probability gap (25 minus 55): "
        f"{youngest_prob - oldest_prob:.2f} percentage points"
    )

    # 4. Count proficiency levels where age changes the decision
    decisions = audit.pivot(
        index="proficiency",
        columns="age",
        values="prediction",
    )

    flips = (
        decisions[25] != decisions[55]
    ).sum()

    print("\nDECISION FLIPS")
    print(
        "Proficiency levels with different decisions at ages "
        f"25 and 55: {flips}/{len(proficiency_levels)}"
    )

    # 5. Verify the direction of the effect at each proficiency
    younger_hired_older_rejected = (
        (decisions[25] == 1) & (decisions[55] == 0)
    ).sum()

    older_hired_younger_rejected = (
        (decisions[25] == 0) & (decisions[55] == 1)
    ).sum()

    print(
        "Young applicant hired, older applicant rejected: "
        f"{younger_hired_older_rejected}"
    )
    print(
        "Older applicant hired, young applicant rejected: "
        f"{older_hired_younger_rejected}"
    )

    return audit


def evaluate_matched_applicants(
    biased_model,
    ordinary_tree,
    scaffolded_tree,
):
    print("\n" + "=" * 65)
    print("MATCHED-APPLICANT EXPLANATION AUDIT")
    print("=" * 65)

    # Identical proficiency, varying age
    ages = [25, 35, 45, 55]
    proficiency_levels = [60, 65, 70, 75, 80, 85, 90, 95, 100]

    profiles = pd.DataFrame([
        {"age": age, "proficiency": proficiency}
        for proficiency in proficiency_levels
        for age in ages
    ])

    # Predictions from the actual biased model
    profiles["actual"] = biased_model.predict(
        profiles[["age", "proficiency"]]
    )

    # Predictions implied by each explanation
    profiles["ordinary_explanation"] = (
        ordinary_tree.predict(profiles[["age", "proficiency"]])
    )

    profiles["scaffolded_explanation"] = (
        scaffolded_tree.predict(profiles[["age", "proficiency"]])
    )

    profiles["actual"] = profiles["actual"].map(
        {0: "Rejected", 1: "Hired"}
    )
    profiles["ordinary_explanation"] = (
        profiles["ordinary_explanation"].map(
            {0: "Rejected", 1: "Hired"}
        )
    )
    profiles["scaffolded_explanation"] = (
        profiles["scaffolded_explanation"].map(
            {0: "Rejected", 1: "Hired"}
        )
    )

    print("\nDECISIONS AT IDENTICAL PROFICIENCY")
    print(profiles.to_string(index=False))

    # Evaluate each explanation against the actual model
    actual = biased_model.predict(
        profiles[["age", "proficiency"]]
    )

    ordinary = ordinary_tree.predict(
        profiles[["age", "proficiency"]]
    )

    scaffolded = scaffolded_tree.predict(
        profiles[["age", "proficiency"]]
    )

    print("\nAGREEMENT WITH THE ACTUAL BIASED MODEL")
    print(
        "Ordinary explanation: "
        f"{np.mean(ordinary == actual):.3f}"
    )
    print(
        "Scaffolded explanation: "
        f"{np.mean(scaffolded == actual):.3f}"
    )

    # Count cases where the actual model distinguishes ages,
    # but the scaffolded explanation gives the same decision.
    actual_differences = 0
    concealed_differences = 0

    for proficiency in proficiency_levels:
        group = profiles[
            profiles["proficiency"] == proficiency
        ]

        actual_decisions = group["actual"].tolist()
        scaffolded_decisions = (
            group["scaffolded_explanation"].tolist()
        )

        if len(set(actual_decisions)) > 1:
            actual_differences += 1

            if len(set(scaffolded_decisions)) == 1:
                concealed_differences += 1

    print("\nAGE-DEPENDENT DECISIONS")
    print(
        "Proficiency levels where the actual model's "
        f"decisions differ by age: {actual_differences}"
    )
    print(
        "Those levels where the scaffolded explanation "
        f"shows no age-based difference: {concealed_differences}"
    )

    if actual_differences:
        print(
            "Fraction of age-dependent levels concealed: "
            f"{concealed_differences / actual_differences:.3f}"
        )
    else:
        print(
            "No age-dependent decision differences were found "
            "at the selected proficiency levels."
        )


def main():
    # --------------------------------------------------
    # 1. Generate synthetic data and train both models
    # --------------------------------------------------
    df = generate_candidates(n=5000)
    fair_model, biased_model = train_models(df)
    evaluate_age_bias(biased_model)

    scaffolded = ScaffoldedRecruitmentModel(
        fair_model=fair_model,
        biased_model=biased_model,
    )

    # Candidate to explain
    candidate = {"age": 42, "proficiency": 90}

    # --------------------------------------------------
    # 2. Generate explanations
    # --------------------------------------------------
    ordinary = explain_candidate(
        biased_model,
        candidate,
        random_state=42,
    )

    attacked = explain_candidate(
        scaffolded,
        candidate,
        is_scaffolded=True,
        random_state=42,
    )

    print("\n" + "=" * 65)
    print("LOCAL EXPLANATIONS")
    print("=" * 65)

    print("\nBIAS MODEL EXPLANATION")
    print(ordinary["rules"])
    print(f"Local fidelity: {ordinary['fidelity']:.3f}")

    print("\nSCAFFOLDED MODEL EXPLANATION")
    print(attacked["rules"])
    print(f"Local fidelity: {attacked['fidelity']:.3f}")

    evaluate_matched_applicants(
    biased_model=biased_model,
    ordinary_tree=ordinary["tree"],
    scaffolded_tree=attacked["tree"],
    )

    # --------------------------------------------------
    # 3. Generate a shared validation neighbourhood
    # --------------------------------------------------
    rng = np.random.default_rng(123)
    n_validation = 1000

    validation = pd.DataFrame({
        "age": np.clip(
            rng.normal(candidate["age"], 8, n_validation).round(),
            21,
            60,
        ).astype(int),
        "proficiency": np.clip(
            rng.normal(
                candidate["proficiency"], 12, n_validation
            ),
            0,
            100,
        ),
    })

    # --------------------------------------------------
    # 4. Compare explanation predictions with the real
    #    biased model on identical validation profiles
    # --------------------------------------------------
    actual_predictions = biased_model.predict(
        validation[["age", "proficiency"]]
    )

    ordinary_predictions = ordinary["tree"].predict(validation)
    scaffolded_predictions = attacked["tree"].predict(validation)

    print("\n" + "=" * 65)
    print("MATCHED EXPLANATION EVALUATION")
    print("=" * 65)

    results = {}

    for name, predictions in [
        ("Ordinary explanation", ordinary_predictions),
        ("Scaffolded explanation", scaffolded_predictions),
    ]:
        agreement = accuracy_score(
            actual_predictions, predictions
        )

        balanced = balanced_accuracy_score(
            actual_predictions, predictions
        )

        results[name] = {
            "agreement": agreement,
            "balanced_agreement": balanced,
        }

        print(f"\n{name}")
        print(
            f"Agreement with biased model: {agreement:.3f}"
        )
        print(
            f"Balanced agreement: {balanced:.3f}"
        )

    # --------------------------------------------------
    # 5. Compare prediction distributions
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("PREDICTION DISTRIBUTIONS")
    print("=" * 65)

    distributions = {
        "Actual biased model": actual_predictions,
        "Ordinary explanation": ordinary_predictions,
        "Scaffolded explanation": scaffolded_predictions,
    }

    for name, predictions in distributions.items():
        print(f"\n{name}:")

        distribution = (
            pd.Series(predictions)
            .value_counts(normalize=True)
            .sort_index()
            .rename(index={0: "Rejected", 1: "Hired"})
        )

        print(distribution.to_string())

    # --------------------------------------------------
    # 6. Measure disagreement between the explanations
    # --------------------------------------------------
    disagreement = np.mean(
        ordinary_predictions != scaffolded_predictions
    )

    print("\n" + "=" * 65)
    print("EXPLANATION DISAGREEMENT")
    print("=" * 65)
    print(
        "Ordinary vs scaffolded disagreement: "
        f"{disagreement:.3f}"
    )

    # --------------------------------------------------
    # 7. Report the probe route separately
    # --------------------------------------------------
    probe_predictions = scaffolded.predict(
        validation,
        is_probe=True,
    )

    probe_agreement = accuracy_score(
        actual_predictions, probe_predictions
    )

    print("\n" + "=" * 65)
    print("PROBE ROUTE DIAGNOSTIC")
    print("=" * 65)
    print(
        "Probe predictions agreeing with biased model: "
        f"{probe_agreement:.3f}"
    )

    print(
        "\nNote: the probe route is explicitly selected using "
        "is_probe=True. This experiment demonstrates the "
        "mechanism in a controlled synthetic setting; it does "
        "not demonstrate automatic detection of explanation queries."
    )


if __name__ == "__main__":
    main()