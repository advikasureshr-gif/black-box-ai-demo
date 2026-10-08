import sys
from pathlib import Path

import numpy as np

# Add project root to Python's import path.
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

        result = detector.analyze(
            logger.get_history()
        )

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
    n_sessions=200,
):
    rng = np.random.default_rng(12345)

    false_positives = 0

    for _ in range(n_sessions):
        tasks = []

        for _ in range(8):
            tasks.append({
                "cpu_required": int(
                    rng.integers(1, 17)
                ),
                "memory_required": int(
                    rng.integers(1, 33)
                ),
                "priority": int(
                    rng.integers(1, 11)
                ),
                "deadline": int(
                    rng.integers(20, 301)
                ),
                "queue_length": int(
                    rng.integers(0, 21)
                ),
            })

        result = run_session(
            detector,
            scheduler,
            tasks,
        )

        if result["detected"]:
            false_positives += 1

    return {
        "false_positive_rate": (
            false_positives / n_sessions
        ),
        "false_positives": false_positives,
        "total_sessions": n_sessions,
    }


def evaluate_probe_type(
    detector,
    scheduler,
    feature,
    values,
    base_task,
    n_sessions=200,
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
        mean_query = float(
            np.mean(detection_queries)
        )
        min_query = int(
            np.min(detection_queries)
        )
        max_query = int(
            np.max(detection_queries)
        )
    else:
        mean_query = None
        min_query = None
        max_query = None

    return {
        "detection_rate": detection_rate,
        "detections": detections,
        "total_sessions": n_sessions,
        "mean_detection_query": mean_query,
        "min_detection_query": min_query,
        "max_detection_query": max_query,
    }


PROBES = {
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


def run_evaluation():
    """
    Run the complete probe-generalization evaluation.

    Returns a structured dictionary so that the final
    run_all_evaluations.py script can reuse these results.
    """

    scheduler = TaskScheduler()

    detector = ProbeDetector(
        probe_threshold=PROBE_THRESHOLD
    )

    normal_result = evaluate_normal_sessions(
        detector=detector,
        scheduler=scheduler,
        n_sessions=NORMAL_SESSIONS,
    )

    probe_results = {}

    for probe_name, config in PROBES.items():
        probe_results[probe_name] = evaluate_probe_type(
            detector=detector,
            scheduler=scheduler,
            feature=config["feature"],
            values=config["values"],
            base_task=config["base_task"],
            n_sessions=PROBE_SESSIONS,
        )

    results = {
        "configuration": {
            "normal_sessions": NORMAL_SESSIONS,
            "probe_sessions": PROBE_SESSIONS,
            "probe_threshold": PROBE_THRESHOLD,
        },
        "normal_interaction": normal_result,
        "probe_generalization": probe_results,
    }

    return results


def print_results(results):
    print("=" * 70)
    print("PROBE GENERALIZATION EVALUATION")
    print("=" * 70)

    config = results["configuration"]

    print()
    print("Configuration")
    print("-" * 70)

    print(
        f"Normal sessions : "
        f"{config['normal_sessions']}"
    )

    print(
        f"Sessions/probe  : "
        f"{config['probe_sessions']}"
    )

    print(
        f"Probe threshold : "
        f"{config['probe_threshold']}"
    )

    normal = results["normal_interaction"]

    print()
    print("Normal Interaction")
    print("-" * 70)

    print(
        f"False-positive rate : "
        f"{normal['false_positive_rate']:.3f} "
        f"({normal['false_positives']}/"
        f"{normal['total_sessions']})"
    )

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

    for probe_name, result in results[
        "probe_generalization"
    ].items():

        mean_query = (
            f"{result['mean_detection_query']:.2f}"
            if result["mean_detection_query"] is not None
            else "N/A"
        )

        min_query = (
            str(result["min_detection_query"])
            if result["min_detection_query"] is not None
            else "N/A"
        )

        max_query = (
            str(result["max_detection_query"])
            if result["max_detection_query"] is not None
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
    results = run_evaluation()
    print_results(results)