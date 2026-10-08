import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.scheduler import TaskScheduler
from src.scheduler_data import FEATURES
from src.query_logger import QueryLogger
from src.probe_detector import ProbeDetector


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_NORMAL_SESSIONS = 30
NUM_PROBE_SESSIONS = 30

PROBE_THRESHOLD = 0.60

scheduler = TaskScheduler()

detector = ProbeDetector(
    window_size=8,
    probe_threshold=PROBE_THRESHOLD,
)

rng = np.random.default_rng(12345)


# ==========================================================
# HELPERS
# ==========================================================

def random_task():
    """
    Generate a normal, independently sampled task.
    """

    return {
        "cpu_required": int(rng.integers(1, 17)),
        "memory_required": int(rng.integers(1, 33)),
        "priority": int(rng.integers(1, 11)),
        "deadline": int(rng.integers(20, 301)),
        "queue_length": int(rng.integers(0, 21)),
    }


def log_task(logger, task):
    """
    Send a task through the scheduler and record it.
    """

    decision = scheduler.predict(task)[0]

    logger.log_query(
        task=task,
        decision=decision,
        explanation_requested=True,
        explanation_released=True,
    )

    return decision


def run_normal_session():
    """
    A normal user submits unrelated tasks.
    """

    logger = QueryLogger()

    for _ in range(8):
        task = random_task()
        log_task(logger, task)

    return logger.get_history()


def run_probe_session():
    """
    A probing user systematically varies memory while
    keeping the remaining task properties fixed.
    """

    logger = QueryLogger()

    base_task = {
        "cpu_required": 4,
        "priority": 5,
        "deadline": 200,
        "queue_length": 5,
    }

    for memory in range(6, 14):

        task = {
            **base_task,
            "memory_required": memory,
        }

        log_task(logger, task)

    return logger.get_history()


def evaluate_session(history):
    """
    Evaluate detector behavior after every query.

    Returns:
        detected
        detection_query
        explanations_exposed
        final_score
    """

    detected = False
    detection_query = None
    explanations_exposed = 0
    final_score = 0.0

    for i in range(1, len(history) + 1):

        result = detector.analyze(
            history[:i]
        )

        final_score = result["probe_score"]

        if result["is_probe"]:

            detected = True
            detection_query = i

            # All previous requests that were processed
            # before detection represent potentially exposed
            # explanations.
            explanations_exposed = i - 1

            break

        explanations_exposed = i

    return {
        "detected": detected,
        "detection_query": detection_query,
        "explanations_exposed": explanations_exposed,
        "final_score": final_score,
    }


# ==========================================================
# NORMAL SESSIONS
# ==========================================================

normal_results = []

for session_id in range(NUM_NORMAL_SESSIONS):

    history = run_normal_session()

    result = evaluate_session(history)

    normal_results.append(result)


# ==========================================================
# PROBE SESSIONS
# ==========================================================

probe_results = []

for session_id in range(NUM_PROBE_SESSIONS):

    history = run_probe_session()

    result = evaluate_session(history)

    probe_results.append(result)


# ==========================================================
# METRICS
# ==========================================================

normal_false_positives = sum(
    result["detected"]
    for result in normal_results
)

probe_true_positives = sum(
    result["detected"]
    for result in probe_results
)

false_positive_rate = (
    normal_false_positives / NUM_NORMAL_SESSIONS
)

detection_rate = (
    probe_true_positives / NUM_PROBE_SESSIONS
)


detected_probe_latencies = [
    result["detection_query"]
    for result in probe_results
    if result["detected"]
]


exposed_explanations = [
    result["explanations_exposed"]
    for result in probe_results
]


if detected_probe_latencies:
    mean_detection_latency = np.mean(
        detected_probe_latencies
    )
else:
    mean_detection_latency = None


mean_explanations_exposed = np.mean(
    exposed_explanations
)


# ==========================================================
# REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("PROBE DETECTOR EVALUATION")
print("=" * 60)

print("\nConfiguration")
print("-" * 60)
print(f"Normal sessions : {NUM_NORMAL_SESSIONS}")
print(f"Probe sessions  : {NUM_PROBE_SESSIONS}")
print(f"Probe threshold : {PROBE_THRESHOLD}")
print(f"Queries/session : 8")


print("\nDetection Performance")
print("-" * 60)

print(
    f"Probe detection rate : "
    f"{detection_rate:.3f} "
    f"({probe_true_positives}/{NUM_PROBE_SESSIONS})"
)

print(
    f"False-positive rate   : "
    f"{false_positive_rate:.3f} "
    f"({normal_false_positives}/{NUM_NORMAL_SESSIONS})"
)


print("\nDetection Latency")
print("-" * 60)

if mean_detection_latency is not None:

    print(
        f"Mean detection query : "
        f"{mean_detection_latency:.2f}"
    )

    print(
        f"Minimum              : "
        f"{min(detected_probe_latencies)}"
    )

    print(
        f"Maximum              : "
        f"{max(detected_probe_latencies)}"
    )

else:

    print("No probes were detected.")


print("\nExplanation Exposure")
print("-" * 60)

print(
    f"Mean explanations exposed before detection: "
    f"{mean_explanations_exposed:.2f}"
)


print("\n")
print("=" * 60)