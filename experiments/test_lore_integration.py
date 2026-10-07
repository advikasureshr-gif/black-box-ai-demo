import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks

from lore_sa.lore import TabularGeneticGeneratorLore
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearnBBox


# --------------------------------------------------
# 1. Generate representative scheduling data
# --------------------------------------------------

data = generate_tasks(
    n=1000,
    random_state=42
)

print("Dataset:")
print(data.head())

print("\nVM distribution:")
print(
    data["assigned_vm"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 2. Create the BLACK-BOX ML scheduler
# --------------------------------------------------

scheduler = TaskScheduler()

print("\nML scheduler created.")


# --------------------------------------------------
# 3. Wrap scheduler for LOREsa
# --------------------------------------------------

bbox = sklearnBBox(scheduler)


# --------------------------------------------------
# 4. Describe scheduler input/output to LOREsa
# --------------------------------------------------

dataset = TabularDataset(
    data,
    class_name="assigned_vm"
)


# --------------------------------------------------
# 5. Create LOREsa explainer
# --------------------------------------------------

explainer = TabularGeneticGeneratorLore(
    bbox,
    dataset
)


# --------------------------------------------------
# 6. Pick a task
# --------------------------------------------------

instance = (
    data
    .drop(columns=["assigned_vm"])
    .iloc[0]
    .values
)


print("\n========================================")
print("TASK TO EXPLAIN")
print("========================================")

print(instance)


# --------------------------------------------------
# 7. Ask the black-box scheduler
# --------------------------------------------------

prediction = scheduler.predict(
    instance.reshape(1, -1)
)[0]

print("\nML Scheduler decision:")
print(prediction)


# --------------------------------------------------
# 8. Generate LOREsa explanation
# --------------------------------------------------

print("\nGenerating LOREsa explanation...")

explanation = explainer.explain(
    instance,
    num_instances=500
)


# --------------------------------------------------
# 9. Display explanation
# --------------------------------------------------

print("\n========================================")
print("LOREsa EXPLANATION")
print("========================================")

print("\nRule:")
print(explanation["rule"])

print("\nCounterfactuals:")
for counterfactual in explanation["counterfactuals"]:
    print(counterfactual)

print("\nFidelity:")
print(explanation["fidelity"])

print("\nFeature importances:")
for feature, importance in explanation["feature_importances"]:
    print(
        f"{feature}: {importance:.4f}"
    )