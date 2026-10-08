import sys
from pathlib import Path

import numpy as np

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler import TaskScheduler
from src.query_logger import QueryLogger
from src.probe_detector import ProbeDetector
from src.explanation_guard import ExplanationGuard


PROBE_THRESHOLD = 0.60
SESSIONS_PER_PROBE = 30


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


def make_task(base_task, feature, value):
    task = dict(base_task)
    task[feature] = value
    return task


def run_unblocked_session(
    scheduler,
    feature,
    values,
    base_task,
):
    """
    Unprotected baseline.

    Every requested explanation is released.
    """

    explanations_exposed = 0
    decisions = []

    for value in values:
        task = make_task(
            base_task,
            feature,
            value,
        )

        decision = scheduler.predict(task)[0]

        decisions.append(decision)
        explanations_exposed += 1

    return {
        "queries": len(list(values)),
        "decisions": decisions,
        "explanations_exposed": explanations_exposed,
    }


def run_blocked_session(
    scheduler,
    detector,
    feature,
    values,
    base_task,
):
    """
    Protected system.

    The scheduler always returns a decision.
    The guard controls whether the explanation is released.
    """

    logger = QueryLogger()

    explanations_exposed = 0
    decisions = []
    detection_query = None

    guard = ExplanationGuard(
        detector=detector
    )

    for query_number, value in enumerate(
        values,
        start=1,
    ):
        task = make_task(
            base_task,
            feature,
            value,
        )

        decision = scheduler.predict(task)[0]
        decisions.append(decision)

        logger.log_query(
            task=task,
            decision=decision,
            explanation_requested=True,
            explanation_released=False,
        )

        guard_result = guard.evaluate(
            logger.get_history()
        )

        if guard_result["explanation_allowed"]:
            explanations_exposed += 1

            logger.history[-1][
                "explanation_released"
            ] = True

        elif detection_query is None:
            detection_query = query_number

    return {
        "queries": len(list(values)),
        "decisions": decisions,
        "explanations_exposed": explanations_exposed,
        "detection_query": detection_query,
    }


def evaluate_probe(
    scheduler,
    detector,
    feature,
    values,
    base_task,
    n_sessions,
):
    unblocked_exposures = []
    blocked_exposures = []
    detection_queries = []

    for _ in range(n_sessions):

        unblocked = run_unblocked_session(
            scheduler=scheduler,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        blocked = run_blocked_session(
            scheduler=scheduler,
            detector=detector,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        unblocked_exposures.append(
            unblocked["explanations_exposed"]
        )

        blocked_exposures.append(
            blocked["explanations_exposed"]
        )

        if blocked["detection_query"] is not None:
            detection_queries.append(
                blocked["detection_query"]
            )

    mean_unblocked = float(
        np.mean(unblocked_exposures)
    )

    mean_blocked = float(
        np.mean(blocked_exposures)
    )

    mean_detection = (
        float(np.mean(detection_queries))
        if detection_queries
        else None
    )

    exposure_reduction = (
        1.0
        - (mean_blocked / mean_unblocked)
    )

    return {
        "unblocked_explanations": mean_unblocked,
        "blocked_explanations": mean_blocked,
        "exposure_reduction": exposure_reduction,
        "mean_detection_query": mean_detection,
        "detections": len(detection_queries),
        "total_sessions": n_sessions,
    }


def run_evaluation():
    """
    Run the complete explanation-exposure evaluation.

    Returns structured results for the final evaluation runner.
    """

    scheduler = TaskScheduler()

    detector = ProbeDetector(
        probe_threshold=PROBE_THRESHOLD
    )

    probe_results = {}

    for probe_name, config in PROBES.items():

        probe_results[probe_name] = evaluate_probe(
            scheduler=scheduler,
            detector=detector,
            feature=config["feature"],
            values=config["values"],
            base_task=config["base_task"],
            n_sessions=SESSIONS_PER_PROBE,
        )

    mean_unblocked = float(
        np.mean([
            result["unblocked_explanations"]
            for result in probe_results.values()
        ])
    )

    mean_blocked = float(
        np.mean([
            result["blocked_explanations"]
            for result in probe_results.values()
        ])
    )

    overall_reduction = (
        1.0
        - (mean_blocked / mean_unblocked)
    )

    return {
        "configuration": {
            "sessions_per_probe": SESSIONS_PER_PROBE,
            "probe_threshold": PROBE_THRESHOLD,
        },
        "probe_results": probe_results,
        "overall": {
            "mean_unblocked_explanations": mean_unblocked,
            "mean_blocked_explanations": mean_blocked,
            "overall_exposure_reduction": overall_reduction,
        },
    }


def print_results(results):

    print("=" * 70)
    print("EXPLANATION EXPOSURE EVALUATION")
    print("=" * 70)

    config = results["configuration"]

    print()
    print("Configuration")
    print("-" * 70)

    print(
        f"Sessions/probe   : "
        f"{config['sessions_per_probe']}"
    )

    print(
        f"Probe threshold  : "
        f"{config['probe_threshold']}"
    )

    print()
    print("Comparison")
    print("-" * 70)

    print(
        f"{'Probe Type':<18}"
        f"{'Unblocked':<14}"
        f"{'Blocked':<14}"
        f"{'Reduction':<14}"
        f"{'Detection':<14}"
    )

    print("-" * 70)

    for probe_name, result in results[
        "probe_results"
    ].items():

        reduction_percent = (
            result["exposure_reduction"] * 100
        )

        detection = (
            f"{result['mean_detection_query']:.2f}"
            if result["mean_detection_query"] is not None
            else "N/A"
        )

        print(
            f"{probe_name:<18}"
            f"{result['unblocked_explanations']:<14.2f}"
            f"{result['blocked_explanations']:<14.2f}"
            f"{reduction_percent:<14.1f}%"
            f"{detection:<14}"
        )

    print()
    print("=" * 70)

    overall = results["overall"]

    print()
    print("Overall")
    print("-" * 70)

    print(
        f"Mean explanations exposed "
        f"(unblocked): "
        f"{overall['mean_unblocked_explanations']:.2f}"
    )

    print(
        f"Mean explanations exposed "
        f"(blocked):   "
        f"{overall['mean_blocked_explanations']:.2f}"
    )

    print(
        f"Overall exposure reduction: "
        f"{overall['overall_exposure_reduction'] * 100:.1f}%"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    results = run_evaluation()
    print_results(results)