
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def explain_candidate(
    model,
    candidate,
    n_samples=1000,
    random_state=42,
    is_scaffolded=False,
):
    rng = np.random.default_rng(random_state)

    # Generate a local neighbourhood around the candidate
    ages = np.clip(
        rng.normal(candidate["age"], 8, n_samples).round(),
        21, 60
    ).astype(int)

    proficiency = np.clip(
        rng.normal(candidate["proficiency"], 12, n_samples),
        0, 100
    )

    neighbourhood = pd.DataFrame({
        "age": ages,
        "proficiency": proficiency,
    })

    # Query the model.
    # For this initial controlled test, mark generated points as probes.
    if is_scaffolded:
        predictions = model.predict(neighbourhood, is_probe=True)
    else:
        predictions = model.predict(neighbourhood)

    # Fit a small, interpretable decision tree
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=random_state,
    )

    tree.fit(neighbourhood, predictions)

    rules = export_text(
        tree,
        feature_names=["age", "proficiency"],
        decimals=2,
    )

    fidelity = tree.score(neighbourhood, predictions)

    return {
        "rules": rules,
        "fidelity": fidelity,
        "neighbourhood": neighbourhood,
        "predictions": predictions,
        "tree": tree,
    }
