import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler import TaskScheduler
from src.query_logger import QueryLogger
from src.probe_detector import ProbeDetector


PROBE_THRESHOLD = 0.60
NORMAL_SESSIONS = 200
PROBE_SESSIONS = 200


def make_task(base_task, feature, value):
    task = dict(base_task)
    task[feature] = value
    return task


def run_session(detector, scheduler, tasks):
    logger = QueryLogger()

    detection_query = None
    final_result = None

    for query_number, task in enumerate(tasks, start=1):
        decision = scheduler.predict(task)[0]

        logger.log_query(
            task=task,
            decision=decision,
            explanation_requested=True,
            explanation_released=False,
        )

        result = detector.analyze(logger.get_history())
        final_result = result

        if result["is_probe"] and detection_query is None:
            detection_query = query_number

    return {
        "detected": detection_query is not None,
        "detection_query": detection_query,
        "final_result": final_result,
    }


def run_feature_probe(
    detector,
    scheduler,
    feature,
    values,
    base_task,
):
    tasks = [
        make_task(base_task, feature, value)
        for value in values
    ]

    return run_session(
        detector,
        scheduler,
        tasks,
    )


def evaluate_normal_sessions(
    detector,
    scheduler,
    n_sessions=30,
):
    rng = np.random.default_rng(12345)

    false_positives = 0

    for _ in range(n_sessions):
        tasks = []

        for _ in range(8):
            task = {
                "cpu_required": int(rng.integers(1, 17)),
                "memory_required": int(rng.integers(1, 33)),
                "priority": int(rng.integers(1, 11)),
                "deadline": int(rng.integers(20, 301)),
                "queue_length": int(rng.integers(0, 21)),
            }

            tasks.append(task)

        result = run_session(
            detector,
            scheduler,
            tasks,
        )

        if result["detected"]:
            false_positives += 1

    return false_positives / n_sessions


def evaluate_probe_type(
    detector,
    scheduler,
    feature,
    values,
    base_task,
    n_sessions=30,
):
    detection_queries = []
    detections = 0

    for _ in range(n_sessions):
        result = run_feature_probe(
            detector=detector,
            scheduler=scheduler,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        if result["detected"]:
            detections += 1
            detection_queries.append(
                result["detection_query"]
            )

    detection_rate = detections / n_sessions

    if detection_queries:
        mean_query = float(np.mean(detection_queries))
        min_query = int(np.min(detection_queries))
        max_query = int(np.max(detection_queries))
    else:
        mean_query = None
        min_query = None
        max_query = None

    return {
        "detection_rate": detection_rate,
        "mean_query": mean_query,
        "min_query": min_query,
        "max_query": max_query,
    }


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

scheduler = TaskScheduler()


# ----------------------------------------------------------------------
# Probe configurations
#
# Each probe varies ONE feature while keeping the remaining task
# properties fixed.
#
# Importantly, each configuration is chosen so that the scan actually
# crosses a scheduler decision boundary.
# ----------------------------------------------------------------------

PROBES = {

    # Expected transition:
    #
    # memory 6 ... 11  -> VM-1
    # memory 12 ... 13 -> VM-3
    #
    "Memory scan": {
        "feature": "memory_required",

        "values": range(6, 14),

        "base_task": {
            "cpu_required": 4,
            "memory_required": 16,
            "priority": 5,
            "deadline": 200,
            "queue_length": 5,
        },
    },

    # With memory >= 12 and cpu < 7, the scheduling rule becomes:
    #
    # deadline <= 120 -> VM-2
    # deadline > 120  -> VM-3
    #
    "Deadline scan": {
        "feature": "deadline",

        "values": range(80, 160, 10),

        "base_task": {
            "cpu_required": 4,
            "memory_required": 16,
            "priority": 5,
            "deadline": 200,
            "queue_length": 5,
        },
    },

    # This deliberately activates the queue-dependent branch:
    #
    # if cpu >= 12 or memory >= 24:
    #     queue <= 8 -> VM-3
    #     queue > 8  -> VM-4
    #
    # Therefore queue 4 ... 12 crosses the boundary.
    #
    "Queue scan": {
        "feature": "queue_length",

        "values": range(4, 13),

        "base_task": {
            "cpu_required": 13,
            "memory_required": 20,
            "priority": 5,
            "deadline": 200,
            "queue_length": 5,
        },
    },
}


def main():

    detector = ProbeDetector(
        probe_threshold=PROBE_THRESHOLD
    )

    print("=" * 70)
    print("PROBE GENERALIZATION EVALUATION")
    print("=" * 70)

    print()
    print("Configuration")
    print("-" * 70)

    print(
        f"Normal sessions : {NORMAL_SESSIONS}"
    )

    print(
        f"Sessions/probe  : {PROBE_SESSIONS}"
    )

    print(
        f"Probe threshold : {PROBE_THRESHOLD}"
    )

    # --------------------------------------------------------------
    # Normal interaction
    # --------------------------------------------------------------

    false_positive_rate = evaluate_normal_sessions(
        detector=detector,
        scheduler=scheduler,
        n_sessions=NORMAL_SESSIONS,
    )

    false_positive_count = round(
        false_positive_rate * NORMAL_SESSIONS
    )

    print()
    print("Normal Interaction")
    print("-" * 70)

    print(
        f"False-positive rate : "
        f"{false_positive_rate:.3f} "
        f"({false_positive_count}/{NORMAL_SESSIONS})"
    )

    # --------------------------------------------------------------
    # Probe generalization
    # --------------------------------------------------------------

    print()
    print("Probe Generalization")
    print("-" * 70)

    print(
        f"{'Probe Type':<18}"
        f"{'Detection':<18}"
        f"{'Mean Query':<15}"
        f"{'Min':<8}"
        f"{'Max':<8}"
    )

    print("-" * 70)

    for probe_name, config in PROBES.items():

        result = evaluate_probe_type(
            detector=detector,
            scheduler=scheduler,
            feature=config["feature"],
            values=config["values"],
            base_task=config["base_task"],
            n_sessions=PROBE_SESSIONS,
        )

        if result["mean_query"] is not None:
            mean_query = f"{result['mean_query']:.2f}"
        else:
            mean_query = "N/A"

        min_query = (
            str(result["min_query"])
            if result["min_query"] is not None
            else "N/A"
        )

        max_query = (
            str(result["max_query"])
            if result["max_query"] is not None
            else "N/A"
        )

        print(
            f"{probe_name:<18}"
            f"{result['detection_rate']:<18.3f}"
            f"{mean_query:<15}"
            f"{min_query:<8}"
            f"{max_query:<8}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()