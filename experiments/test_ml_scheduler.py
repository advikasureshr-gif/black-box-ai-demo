import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.scheduler import TaskScheduler
from src.scheduler_data import (
    generate_tasks,
    FEATURES,
)


print("Generating evaluation dataset...")

data = generate_tasks(
    n=2000,
    random_state=123,
)

X = data[FEATURES]
y = data["assigned_vm"]


# --------------------------------------------------
# Train/test split for evaluating the learned model
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Train scheduler
# --------------------------------------------------

scheduler = TaskScheduler()


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = scheduler.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)


print("\n========== ML SCHEDULER ==========")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
    )
)


# --------------------------------------------------
# Example predictions
# --------------------------------------------------

print("\n========== EXAMPLE TASKS ==========")

examples = X_test.head(5)

example_predictions = scheduler.predict(
    examples
)

for (_, task), prediction in zip(
    examples.iterrows(),
    example_predictions,
):

    print("\nTask:")
    print(task.to_dict())

    print(
        "Selected VM:",
        prediction,
    )