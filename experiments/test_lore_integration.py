import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import pandas as pd

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks

from lore_sa.lore import TabularGeneticGeneratorLore
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearnBBox


# --------------------------------------------------
# 1. Generate scheduler dataset
# --------------------------------------------------

data = generate_tasks(
    n=500,
    random_state=42
)

print("Dataset:")
print(data.head())

print("\nVM distribution:")
print(data["assigned_vm"].value_counts().sort_index())


# --------------------------------------------------
# 2. Create black-box scheduler
# --------------------------------------------------

scheduler = TaskScheduler()

bbox = sklearnBBox(scheduler)


# --------------------------------------------------
# 3. Tell LOREsa what the dataset looks like
# --------------------------------------------------

dataset = TabularDataset(
    data,
    class_name="assigned_vm"
)


# --------------------------------------------------
# 4. Create LOREsa explainer
# --------------------------------------------------

explainer = TabularGeneticGeneratorLore(
    bbox,
    dataset
)


# --------------------------------------------------
# 5. Select one task to explain
# --------------------------------------------------

instance = data.drop(
    columns=["assigned_vm"]
).iloc[0].values


print("\nInstance being explained:")
print(instance)

print(
    "\nScheduler decision:",
    scheduler.predict(instance.reshape(1, -1))[0]
)


# --------------------------------------------------
# 6. Generate explanation
# --------------------------------------------------

print("\nGenerating LOREsa explanation...")

explanation = explainer.explain(
    instance,
    num_instances=500
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n========== LOREsa RESULT ==========")

print("\nRule:")
print(explanation["rule"])

print("\nCounterfactuals:")
print(explanation["counterfactuals"])

print("\nFidelity:")
print(explanation["fidelity"])

print("\nFeature importances:")
print(explanation["feature_importances"])