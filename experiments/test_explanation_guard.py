import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import TaskScheduler
from src.query_logger import QueryLogger
from src.explanation_guard import ExplanationGuard


scheduler = TaskScheduler()
guard = ExplanationGuard()


# ==========================================================
# NORMAL USER
# ==========================================================

normal_logger = QueryLogger()

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
    {
        "cpu_required": 8,
        "memory_required": 25,
        "priority": 10,
        "deadline": 175,
        "queue_length": 7,
    },
    {
        "cpu_required": 7,
        "memory_required": 23,
        "priority": 8,
        "deadline": 269,
        "queue_length": 1,
    },
]

for task in normal_tasks:

    decision = scheduler.predict(task)[0]

    normal_logger.log_query(
        task=task,
        decision=decision,
        explanation_requested=True,
        explanation_released=True,
    )


# ==========================================================
# PROBE USER
# ==========================================================

probe_logger = QueryLogger()

for memory in range(6, 14):

    task = {
        "cpu_required": 4,
        "memory_required": memory,
        "priority": 5,
        "deadline": 200,
        "queue_length": 5,
    }

    decision = scheduler.predict(task)[0]

    probe_logger.log_query(
        task=task,
        decision=decision,
        explanation_requested=True,
        explanation_released=True,
    )


# ==========================================================
# GUARD EVALUATION
# ==========================================================

normal_result = guard.evaluate(
    normal_logger.get_history()
)

probe_result = guard.evaluate(
    probe_logger.get_history()
)


print("\n========================================")
print("NORMAL USER")
print("========================================")

print("Probe score:", normal_result["probe_score"])
print("Probe detected:", normal_result["is_probe"])
print("Explanation allowed:", normal_result["explanation_allowed"])


print("\n========================================")
print("PROBE USER")
print("========================================")

print("Probe score:", probe_result["probe_score"])
print("Probe detected:", probe_result["is_probe"])
print("Explanation allowed:", probe_result["explanation_allowed"])