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
        "values": list(range(6, 14)),
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
        "values": list(range(80, 160, 10)),
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
        "values": list(range(4, 13)),
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


def find_boundary(values, decisions):
    """
    Find the first observed decision transition.

    Returns None if no transition is visible.
    """

    for i in range(1, len(decisions)):
        if decisions[i] != decisions[i - 1]:
            return {
                "boundary_index": i,
                "before_value": values[i - 1],
                "after_value": values[i],
                "before_decision": decisions[i - 1],
                "after_decision": decisions[i],
            }

    return None


def run_unblocked_probe(
    scheduler,
    feature,
    values,
    base_task,
):
    """
    Unprotected baseline.

    The complete probe sequence is observable.
    """

    decisions = []

    for value in values:
        task = make_task(
            base_task,
            feature,
            value,
        )

        decision = scheduler.predict(task)[0]
        decisions.append(decision)

    boundary = find_boundary(
        list(values),
        decisions,
    )

    return {
        "values_seen": list(values),
        "decisions_seen": decisions,
        "queries_seen": len(list(values)),
        "boundary_observed": boundary is not None,
        "boundary": boundary,
    }


def run_blocked_probe(
    scheduler,
    detector,
    feature,
    values,
    base_task,
):
    """
    Protected system.

    The scheduler continues returning decisions, but once the
    explanation guard blocks access, subsequent information is
    not counted as exposed to the probing user.
    """

    logger = QueryLogger()

    exposed_values = []
    exposed_decisions = []

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

            exposed_values.append(value)
            exposed_decisions.append(decision)

            logger.history[-1][
                "explanation_released"
            ] = True

        elif detection_query is None:
            detection_query = query_number

    boundary = find_boundary(
        exposed_values,
        exposed_decisions,
    )

    return {
        "values_seen": exposed_values,
        "decisions_seen": exposed_decisions,
        "queries_seen": len(exposed_values),
        "boundary_observed": boundary is not None,
        "boundary": boundary,
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
    unblocked_boundary_observed = 0
    blocked_boundary_observed = 0

    detection_queries = []
    exposed_queries = []

    for _ in range(n_sessions):

        unblocked = run_unblocked_probe(
            scheduler=scheduler,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        blocked = run_blocked_probe(
            scheduler=scheduler,
            detector=detector,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        if unblocked["boundary_observed"]:
            unblocked_boundary_observed += 1

        if blocked["boundary_observed"]:
            blocked_boundary_observed += 1

        if blocked["detection_query"] is not None:
            detection_queries.append(
                blocked["detection_query"]
            )

        exposed_queries.append(
            blocked["queries_seen"]
        )

    return {
        "unblocked_boundary_rate": (
            unblocked_boundary_observed / n_sessions
        ),
        "blocked_boundary_rate": (
            blocked_boundary_observed / n_sessions
        ),
        "mean_detection_query": (
            float(np.mean(detection_queries))
            if detection_queries
            else None
        ),
        "mean_exposed_queries": (
            float(np.mean(exposed_queries))
        ),
        "detection_rate": (
            len(detection_queries) / n_sessions
        ),
        "detections": len(detection_queries),
        "total_sessions": n_sessions,
    }


def run_evaluation():
    """
    Run the complete boundary-leakage evaluation.

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

    mean_unblocked_boundary_rate = float(
        np.mean([
            result["unblocked_boundary_rate"]
            for result in probe_results.values()
        ])
    )

    mean_blocked_boundary_rate = float(
        np.mean([
            result["blocked_boundary_rate"]
            for result in probe_results.values()
        ])
    )

    mean_exposed_queries = float(
        np.mean([
            result["mean_exposed_queries"]
            for result in probe_results.values()
        ])
    )

    mean_detection_query = float(
        np.mean([
            result["mean_detection_query"]
            for result in probe_results.values()
            if result["mean_detection_query"] is not None
        ])
    )

    return {
        "configuration": {
            "sessions_per_probe": SESSIONS_PER_PROBE,
            "probe_threshold": PROBE_THRESHOLD,
        },
        "probe_results": probe_results,
        "overall": {
            "mean_unblocked_boundary_rate": (
                mean_unblocked_boundary_rate
            ),
            "mean_blocked_boundary_rate": (
                mean_blocked_boundary_rate
            ),
            "mean_exposed_queries": (
                mean_exposed_queries
            ),
            "mean_detection_query": (
                mean_detection_query
            ),
        },
    }


def print_results(results):

    print("=" * 70)
    print("BOUNDARY LEAKAGE EVALUATION")
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
    print("Boundary Visibility")
    print("-" * 70)

    print(
        f"{'Probe Type':<18}"
        f"{'Unblocked':<16}"
        f"{'Blocked':<16}"
        f"{'Detection':<16}"
        f"{'Exposed':<12}"
    )

    print("-" * 70)

    for probe_name, result in results[
        "probe_results"
    ].items():

        detection = (
            f"{result['mean_detection_query']:.2f}"
            if result["mean_detection_query"] is not None
            else "N/A"
        )

        print(
            f"{probe_name:<18}"
            f"{result['unblocked_boundary_rate']:<16.3f}"
            f"{result['blocked_boundary_rate']:<16.3f}"
            f"{detection:<16}"
            f"{result['mean_exposed_queries']:<12.2f}"
        )

    print()
    print("=" * 70)

    print()
    print("Interpretation")
    print("-" * 70)

    print(
        "Unblocked = fraction of sessions where the complete "
        "probe exposed the scheduler's decision transition."
    )

    print(
        "Blocked = fraction of sessions where the decision "
        "transition was visible before explanation access "
        "was restricted."
    )

    print(
        "Exposed = mean number of probe queries for which "
        "explanation access remained available."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    results = run_evaluation()
    print_results(results)