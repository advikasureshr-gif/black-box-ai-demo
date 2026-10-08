import sys
from pathlib import Path

import numpy as np

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler import TaskScheduler
from src.query_logger import QueryLogger
from src.probe_detector import ProbeDetector


# ============================================================
# CONFIGURATION
# ============================================================

NUM_NORMAL_SESSIONS = 200
NUM_PROBE_SESSIONS = 200

PROBE_THRESHOLD = 0.60
QUERIES_PER_SESSION = 8


# ============================================================
# HELPERS
# ============================================================

def random_task(rng):
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


def log_task(scheduler, logger, task):
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


def run_normal_session(scheduler, rng):
    """
    A normal user submits unrelated tasks.
    """

    logger = QueryLogger()

    for _ in range(QUERIES_PER_SESSION):
        task = random_task(rng)

        log_task(
            scheduler=scheduler,
            logger=logger,
            task=task,
        )

    return logger.get_history()


def run_probe_session(scheduler):
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

        log_task(
            scheduler=scheduler,
            logger=logger,
            task=task,
        )

    return logger.get_history()


def evaluate_session(history, detector):
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

            # All requests processed before detection
            # represent potentially exposed explanations.
            explanations_exposed = i - 1

            break

        explanations_exposed = i

    return {
        "detected": detected,
        "detection_query": detection_query,
        "explanations_exposed": explanations_exposed,
        "final_score": final_score,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():
    """
    Run the complete probe detector evaluation.

    Returns structured results for the final evaluation runner.
    """

    scheduler = TaskScheduler()

    detector = ProbeDetector(
        window_size=8,
        probe_threshold=PROBE_THRESHOLD,
    )

    rng = np.random.default_rng(12345)

    # --------------------------------------------------------
    # Normal sessions
    # --------------------------------------------------------

    normal_results = []

    for _ in range(NUM_NORMAL_SESSIONS):

        history = run_normal_session(
            scheduler=scheduler,
            rng=rng,
        )

        result = evaluate_session(
            history=history,
            detector=detector,
        )

        normal_results.append(result)

    # --------------------------------------------------------
    # Probe sessions
    # --------------------------------------------------------

    probe_results = []

    for _ in range(NUM_PROBE_SESSIONS):

        history = run_probe_session(
            scheduler=scheduler,
        )

        result = evaluate_session(
            history=history,
            detector=detector,
        )

        probe_results.append(result)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    normal_false_positives = sum(
        result["detected"]
        for result in normal_results
    )

    probe_true_positives = sum(
        result["detected"]
        for result in probe_results
    )

    false_positive_rate = (
        normal_false_positives
        / NUM_NORMAL_SESSIONS
    )

    detection_rate = (
        probe_true_positives
        / NUM_PROBE_SESSIONS
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

        mean_detection_latency = float(
            np.mean(
                detected_probe_latencies
            )
        )

        minimum_detection_latency = int(
            min(detected_probe_latencies)
        )

        maximum_detection_latency = int(
            max(detected_probe_latencies)
        )

    else:

        mean_detection_latency = None
        minimum_detection_latency = None
        maximum_detection_latency = None

    mean_explanations_exposed = float(
        np.mean(
            exposed_explanations
        )
    )

    return {
        "configuration": {
            "normal_sessions": NUM_NORMAL_SESSIONS,
            "probe_sessions": NUM_PROBE_SESSIONS,
            "probe_threshold": PROBE_THRESHOLD,
            "queries_per_session": QUERIES_PER_SESSION,
        },

        "probe_detection_rate": detection_rate,

        "false_positive_rate": false_positive_rate,

        "probe_true_positives": probe_true_positives,

        "normal_false_positives": normal_false_positives,

        "mean_detection_query": (
            mean_detection_latency
        ),

        "minimum_detection_query": (
            minimum_detection_latency
        ),

        "maximum_detection_query": (
            maximum_detection_latency
        ),

        "mean_explanations_exposed": (
            mean_explanations_exposed
        ),
    }


# ============================================================
# REPORT
# ============================================================

def print_results(results):

    config = results["configuration"]

    print()
    print("=" * 60)
    print("PROBE DETECTOR EVALUATION")
    print("=" * 60)

    print()
    print("Configuration")
    print("-" * 60)

    print(
        f"Normal sessions : "
        f"{config['normal_sessions']}"
    )

    print(
        f"Probe sessions  : "
        f"{config['probe_sessions']}"
    )

    print(
        f"Probe threshold : "
        f"{config['probe_threshold']}"
    )

    print(
        f"Queries/session : "
        f"{config['queries_per_session']}"
    )

    print()
    print("Detection Performance")
    print("-" * 60)

    print(
        f"Probe detection rate : "
        f"{results['probe_detection_rate']:.3f} "
        f"({results['probe_true_positives']}/"
        f"{config['probe_sessions']})"
    )

    print(
        f"False-positive rate  : "
        f"{results['false_positive_rate']:.3f} "
        f"({results['normal_false_positives']}/"
        f"{config['normal_sessions']})"
    )

    print()
    print("Detection Latency")
    print("-" * 60)

    if results["mean_detection_query"] is not None:

        print(
            f"Mean detection query : "
            f"{results['mean_detection_query']:.2f}"
        )

        print(
            f"Minimum              : "
            f"{results['minimum_detection_query']}"
        )

        print(
            f"Maximum              : "
            f"{results['maximum_detection_query']}"
        )

    else:

        print("No probes were detected.")

    print()
    print("Explanation Exposure")
    print("-" * 60)

    print(
        f"Mean explanations exposed before detection: "
        f"{results['mean_explanations_exposed']:.2f}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":

    results = run_evaluation()

    print_results(results)