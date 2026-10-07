import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import TaskScheduler
from src.query_logger import QueryLogger
from src.probe_detector import ProbeDetector


scheduler = TaskScheduler()
detector = ProbeDetector(window_size=8)


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
# PROBE
# ==========================================================

probe_logger = QueryLogger()

# Keep everything fixed while systematically scanning memory.
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
# ANALYSIS
# ==========================================================

normal_result = detector.analyze(
    normal_logger.get_history()
)

print("\n========================================")
print("PROBE QUERY SEQUENCE")
print("========================================")

for i, query in enumerate(probe_logger.get_history(), start=1):
    print(
        f"{i}: "
        f"memory={query['task']['memory_required']} "
        f"→ {query['decision']}"
    )
    
probe_result = detector.analyze(
    probe_logger.get_history()
)


print("\n========================================")
print("NORMAL USER")
print("========================================")

print("Probe score:", normal_result["probe_score"])
print("Is probe:", normal_result["is_probe"])
print("Signals:")

for key, value in normal_result["signals"].items():
    print(f"  {key}: {value}")


print("\n========================================")
print("PROBE USER")
print("========================================")

print("Probe score:", probe_result["probe_score"])
print("Is probe:", probe_result["is_probe"])
print("Signals:")

for key, value in probe_result["signals"].items():
    print(f"  {key}: {value}")