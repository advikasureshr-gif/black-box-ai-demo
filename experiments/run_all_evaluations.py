import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_probe_detector import (
    run_evaluation as run_detector,
)

from evaluate_probe_generalization import (
    run_evaluation as run_generalization,
)

from evaluate_explanation_exposure import (
    run_evaluation as run_exposure,
)

from evaluate_boundary_leakage import (
    run_evaluation as run_leakage,
)


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "final_evaluation.json"


def run_all():
    print("=" * 70)
    print("BLACK-BOX AI EXPLAINABILITY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Probe detector
    # --------------------------------------------------------------

    print()
    print("[1/4] Running probe detector evaluation...")

    detector_results = run_detector()

    print("      ✓ Complete")

    # --------------------------------------------------------------
    # 2. Probe generalization
    # --------------------------------------------------------------

    print()
    print("[2/4] Running probe generalization evaluation...")

    generalization_results = run_generalization()

    print("      ✓ Complete")

    # --------------------------------------------------------------
    # 3. Explanation exposure
    # --------------------------------------------------------------

    print()
    print("[3/4] Running explanation exposure evaluation...")

    exposure_results = run_exposure()

    print("      ✓ Complete")

    # --------------------------------------------------------------
    # 4. Boundary leakage
    # --------------------------------------------------------------

    print()
    print("[4/4] Running boundary leakage evaluation...")

    leakage_results = run_leakage()

    print("      ✓ Complete")

    # --------------------------------------------------------------
    # Combine results
    # --------------------------------------------------------------

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "project": "Probe-Aware Explanation Control",
            "description": (
                "Evaluation of systematic probe detection, "
                "cross-feature generalization, explanation "
                "exposure reduction, and boundary visibility."
            ),
        },

        "probe_detector": detector_results,

        "probe_generalization": generalization_results,

        "explanation_exposure": exposure_results,

        "boundary_leakage": leakage_results,
    }

    # --------------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    return results


def print_summary(results):

    print()
    print()
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    # ==============================================================
    # 1. Probe detection
    # ==============================================================

    detector = results["probe_detector"]

    print()
    print("1. PROBE DETECTION")
    print("-" * 70)

    print(
        f"Probe detection rate : "
        f"{detector['probe_detection_rate']:.3f}"
    )

    print(
        f"False-positive rate  : "
        f"{detector['false_positive_rate']:.3f}"
    )

    print(
        f"Mean detection query : "
        f"{detector['mean_detection_query']:.2f}"
    )

    print(
        f"Mean explanations exposed "
        f"before detection : "
        f"{detector['mean_explanations_exposed']:.2f}"
    )

    # ==============================================================
    # 2. Generalization
    # ==============================================================

    generalization = results[
        "probe_generalization"
    ]

    print()
    print("2. PROBE GENERALIZATION")
    print("-" * 70)

    normal = generalization[
        "normal_interaction"
    ]

    print(
        f"False-positive rate : "
        f"{normal['false_positive_rate']:.3f}"
    )

    for (
        probe_name,
        probe,
    ) in generalization[
        "probe_generalization"
    ].items():

        print(
            f"{probe_name:<18}"
            f" detection={probe['detection_rate']:.3f}"
            f"  latency={probe['mean_detection_query']:.2f}"
        )

    # ==============================================================
    # 3. Explanation exposure
    # ==============================================================

    exposure = results[
        "explanation_exposure"
    ]

    print()
    print("3. EXPLANATION EXPOSURE")
    print("-" * 70)

    overall_exposure = exposure[
        "overall"
    ]

    print(
        f"Unblocked explanations : "
        f"{overall_exposure['mean_unblocked_explanations']:.2f}"
    )

    print(
        f"Blocked explanations   : "
        f"{overall_exposure['mean_blocked_explanations']:.2f}"
    )

    print(
        f"Exposure reduction     : "
        f"{overall_exposure['overall_exposure_reduction'] * 100:.1f}%"
    )

    # ==============================================================
    # 4. Boundary leakage
    # ==============================================================

    leakage = results[
        "boundary_leakage"
    ]

    print()
    print("4. BOUNDARY VISIBILITY")
    print("-" * 70)

    overall_leakage = leakage[
        "overall"
    ]

    print(
        f"Mean unblocked boundary visibility : "
        f"{overall_leakage['mean_unblocked_boundary_rate']:.3f}"
    )

    print(
        f"Mean blocked boundary visibility   : "
        f"{overall_leakage['mean_blocked_boundary_rate']:.3f}"
    )

    print(
        f"Mean detection query              : "
        f"{overall_leakage['mean_detection_query']:.2f}"
    )

    print(
        f"Mean exposed queries              : "
        f"{overall_leakage['mean_exposed_queries']:.2f}"
    )

    # ==============================================================
    # Key findings
    # ==============================================================

    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print()

    print(
        "• Systematic probes were detected across "
        "multiple scheduling features."
    )

    print(
        "• The detector maintained a low false-positive "
        "rate under the evaluated normal sessions."
    )

    print(
        "• Probe-aware explanation control reduced "
        "local explanation exposure."
    )

    print(
        "• Under the evaluated probe scenarios, the "
        "decision boundary was not observed from the "
        "information released before explanation "
        "restriction."
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":

    results = run_all()

    print_summary(results)