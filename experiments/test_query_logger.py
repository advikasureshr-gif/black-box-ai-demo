import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks
from src.query_logger import QueryLogger


data = generate_tasks(n=10, random_state=42)

scheduler = TaskScheduler()
logger = QueryLogger()

for _, row in data.drop(columns=["assigned_vm"]).head(5).iterrows():

    task = row.to_dict()

    decision = scheduler.predict([task])[0]

    logger.log_query(
        task=task,
        decision=decision,
        explanation_requested=True,
        explanation_released=True,
    )


print("\n========================================")
print("QUERY LOGGER TEST")
print("========================================")

print(f"\nTotal queries logged: {len(logger)}")

for i, query in enumerate(logger.get_history(), start=1):
    print(f"\nQuery {i}")
    print("Task:", query["task"])
    print("Decision:", query["decision"])
    print("Explanation requested:", query["explanation_requested"])
    print("Explanation released:", query["explanation_released"])