import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks
from src.lore_explainer import LOREExplainer


# --------------------------------------------------
# Generate reference data
# --------------------------------------------------

data = generate_tasks(
    n=1000,
    random_state=42,
)


# --------------------------------------------------
# Create black-box scheduler
# --------------------------------------------------

scheduler = TaskScheduler()


# --------------------------------------------------
# Create LOREsa adapter
# --------------------------------------------------

explainer = LOREExplainer(
    scheduler=scheduler,
    reference_data=data,
    num_instances=500,
)


# --------------------------------------------------
# Select a task
# --------------------------------------------------

task = (
    data
    .drop(columns=["assigned_vm"])
    .iloc[0]
    .to_dict()
)


print("\n========================================")
print("TASK")
print("========================================")

print(task)


# --------------------------------------------------
# Explain
# --------------------------------------------------

print("\nGenerating explanation...")

result = explainer.explain(
    task
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n========================================")
print("RESULT")
print("========================================")

print("\nPrediction:")
print(result["prediction"])

print("\nRule:")
print(result["rule"])

print("\nCounterfactuals:")

for counterfactual in result[
    "counterfactuals"
]:
    print(counterfactual)

print("\nFidelity:")
print(result["fidelity"])

print("\nFeature importances:")

for feature, importance in result[
    "feature_importances"
]:
    print(
        f"{feature}: {importance:.4f}"
    )