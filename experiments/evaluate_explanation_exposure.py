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


# ----------------------------------------------------------------------
# Probe configurations
# ----------------------------------------------------------------------

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
    Simulate an unrestricted XAI system.

    Every query is assumed to receive a local explanation.
    The detector is not used to restrict access.
    """

    decisions = []
    explanations_exposed = 0

    for value in values:
        task = make_task(
            base_task,
            feature,
            value,
        )

        decision = scheduler.predict(task)[0]

        decisions.append(decision)

        # In the unblocked baseline, every requested
        # explanation is released.
        explanations_exposed += 1

    transitions = sum(
        decisions[i] != decisions[i - 1]
        for i in range(1, len(decisions))
    )

    return {
        "queries": len(values),
        "decisions": decisions,
        "decision_transitions": transitions,
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
    Run the same probe through the proposed
    probe-aware explanation guard.

    The scheduler always returns a decision.
    Only explanation access is restricted.
    """

    logger = QueryLogger()

    decisions = []
    explanations_exposed = 0
    detection_query = None

    for query_number, value in enumerate(values, start=1):

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

        guard = ExplanationGuard(
            detector=detector
        )

        guard_result = guard.evaluate(
            logger.get_history()
        )

        if guard_result["explanation_allowed"]:
            explanations_exposed += 1
            logger.history[-1]["explanation_released"] = True

        if (
            guard_result["is_probe"]
            and detection_query is None
        ):
            detection_query = query_number

    transitions = sum(
        decisions[i] != decisions[i - 1]
        for i in range(1, len(decisions))
    )

    return {
        "queries": len(values),
        "decisions": decisions,
        "decision_transitions": transitions,
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

        # --------------------------------------------------------------
        # Unblocked baseline
        # --------------------------------------------------------------

        unblocked = run_unblocked_session(
            scheduler=scheduler,
            feature=feature,
            values=values,
            base_task=base_task,
        )

        unblocked_exposures.append(
            unblocked["explanations_exposed"]
        )

        # --------------------------------------------------------------
        # Protected system
        # --------------------------------------------------------------

        blocked = run_blocked_session(
            scheduler=scheduler,
            detector=detector,
            feature=feature,
            values=values,
            base_task=base_task,
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

    max_queries = len(list(values))

    exposure_reduction = (
        1.0
        - (mean_blocked / mean_unblocked)
    )

    return {
        "unblocked": mean_unblocked,
        "blocked": mean_blocked,
        "reduction": exposure_reduction,
        "detection_query": mean_detection,
        "detections": len(detection_queries),
        "total_sessions": n_sessions,
        "max_queries": max_queries,
    }


def main():

    scheduler = TaskScheduler()

    detector = ProbeDetector(
        probe_threshold=PROBE_THRESHOLD
    )

    print("=" * 70)
    print("EXPLANATION EXPOSURE EVALUATION")
    print("=" * 70)

    print()
    print("Configuration")
    print("-" * 70)
    print(
        f"Sessions/probe   : {SESSIONS_PER_PROBE}"
    )
    print(
        f"Probe threshold  : {PROBE_THRESHOLD}"
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

    all_results = {}

    for probe_name, config in PROBES.items():

        result = evaluate_probe(
            scheduler=scheduler,
            detector=detector,
            feature=config["feature"],
            values=config["values"],
            base_task=config["base_task"],
            n_sessions=SESSIONS_PER_PROBE,
        )

        all_results[probe_name] = result

        reduction_percent = (
            result["reduction"] * 100
        )

        if result["detection_query"] is not None:
            detection = (
                f"{result['detection_query']:.2f}"
            )
        else:
            detection = "N/A"

        print(
            f"{probe_name:<18}"
            f"{result['unblocked']:<14.2f}"
            f"{result['blocked']:<14.2f}"
            f"{reduction_percent:<14.1f}%"
            f"{detection:<14}"
        )

    print()
    print("=" * 70)

    # --------------------------------------------------------------
    # Overall result
    # --------------------------------------------------------------

    total_unblocked = sum(
        result["unblocked"]
        for result in all_results.values()
    )

    total_blocked = sum(
        result["blocked"]
        for result in all_results.values()
    )

    overall_reduction = (
        1.0
        - (total_blocked / total_unblocked)
    )

    print()
    print("Overall")
    print("-" * 70)

    print(
        f"Mean explanations exposed "
        f"(unblocked): {total_unblocked / len(PROBES):.2f}"
    )

    print(
        f"Mean explanations exposed "
        f"(blocked):   {total_blocked / len(PROBES):.2f}"
    )

    print(
        f"Overall exposure reduction: "
        f"{overall_reduction * 100:.1f}%"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()