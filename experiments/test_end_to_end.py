import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import TaskScheduler
from src.scheduler_data import generate_tasks
from src.explanation_service import ExplanationService


# ==========================================================
# SETUP
# ==========================================================

reference_data = generate_tasks(
    n=1000,
    random_state=42,
)

scheduler = TaskScheduler()

service = ExplanationService(
    scheduler=scheduler,
    reference_data=reference_data,
    num_instances=500,
)


# ==========================================================
# NORMAL USER
# ==========================================================

normal_tasks = [
    {
        "cpu_required": 2,
        "memory_required": 17,
        "priority": 6,
        "deadline": 146,
        "queue_length": 3,
    },
    {
        "cpu_required": 13,
        "memory_required": 32,
        "priority": 4,
        "deadline": 83,
        "queue_length": 15,
    },
    {
        "cpu_required": 11,
        "memory_required": 24,
        "priority": 2,
        "deadline": 45,
        "queue_length": 14,
    },
]


print("\n========================================")
print("NORMAL USER")
print("========================================")

for task in normal_tasks:

    result = service.process_task(task)

    print("\nTask:", task)
    print("Decision:", result["decision"])
    print("Probe score:", result["probe_score"])
    print("Probe detected:", result["is_probe"])
    print(
        "Explanation allowed:",
        result["explanation_allowed"],
    )

    if result["explanation"] is not None:
        print("LOREsa explanation: AVAILABLE")
        print(
            "Fidelity:",
            result["explanation"]["fidelity"],
        )
    else:
        print("LOREsa explanation: BLOCKED")


# ==========================================================
# PROBE USER
# ==========================================================

print("\n\n========================================")
print("PROBE USER")
print("========================================")

for memory in range(6, 14):

    task = {
        "cpu_required": 4,
        "memory_required": memory,
        "priority": 5,
        "deadline": 200,
        "queue_length": 5,
    }

    result = service.process_task(task)

    print(
        f"\nmemory={memory}"
        f" → {result['decision']}"
        f" | probe score={result['probe_score']}"
        f" | blocked={not result['explanation_allowed']}"
    )