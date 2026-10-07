from .probe_detector import ProbeDetector


class ExplanationGuard:
    """
    Controls access to local explanations based on observed
    query behavior.

    The scheduler always remains operational. The guard only
    controls whether the LOREsa explanation is released.
    """

    def __init__(self, detector=None):
        self.detector = detector or ProbeDetector()

    def evaluate(self, history):
        """
        Analyze query history and determine whether an
        explanation should be released.
        """

        result = self.detector.analyze(history)

        explanation_allowed = not result["is_probe"]

        return {
            "explanation_allowed": explanation_allowed,
            "probe_score": result["probe_score"],
            "is_probe": result["is_probe"],
            "signals": result["signals"],
        }