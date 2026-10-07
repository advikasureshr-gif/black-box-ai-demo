import numpy as np

from .scheduler_data import FEATURES


class ProbeDetector:
    """
    Detects systematic decision-boundary probing from a sequence
    of scheduler queries.

    Signals:
        - systematic one-feature variation
        - monotonic scanning of that feature
        - scheduler decision transitions
        - repeated queries
        - query volume

    The detector is intentionally interpretable and heuristic.
    It is a proposed layer around the scheduler and LOREsa,
    not part of LOREsa itself.
    """

    def __init__(
        self,
        window_size=8,
        variation_threshold=3,
        decision_change_threshold=1,
        probe_threshold=0.60,
    ):
        self.window_size = window_size
        self.variation_threshold = variation_threshold
        self.decision_change_threshold = decision_change_threshold
        self.probe_threshold = probe_threshold

    def _is_monotonic(self, values):
        """
        Check whether values are consistently increasing
        or decreasing.
        """

        if len(values) < 3:
            return False

        increasing = all(
            values[i] <= values[i + 1]
            for i in range(len(values) - 1)
        )

        decreasing = all(
            values[i] >= values[i + 1]
            for i in range(len(values) - 1)
        )

        return increasing or decreasing

    def _find_decision_transitions(self, decisions):
        """
        Return the positions where the scheduler decision changes.
        """

        transitions = []

        for i in range(1, len(decisions)):
            if decisions[i] != decisions[i - 1]:
                transitions.append(i)

        return transitions

    def analyze(self, history):

        if len(history) < 2:
            return {
                "probe_score": 0.0,
                "is_probe": False,
                "signals": {
                    "insufficient_history": True
                },
            }

        queries = history[-self.window_size:]

        feature_values = {
            feature: []
            for feature in FEATURES
        }

        decisions = []

        for query in queries:

            task = query["task"]

            for feature in FEATURES:
                feature_values[feature].append(
                    task[feature]
                )

            decisions.append(query["decision"])

        # ==================================================
        # Signal 1: systematic feature variation
        # ==================================================

        systematic_features = []

        for feature in FEATURES:

            values = feature_values[feature]

            if len(set(values)) < self.variation_threshold:
                continue

            # Count how many other features remain stable.
            stable_other_features = 0

            for other_feature in FEATURES:

                if other_feature == feature:
                    continue

                other_values = feature_values[other_feature]

                if len(set(other_values)) <= 2:
                    stable_other_features += 1

            # We want the candidate feature to vary while
            # most other dimensions remain stable.
            if stable_other_features >= len(FEATURES) - 2:
                systematic_features.append(feature)

        # ==================================================
        # Signal 2: monotonic scanning
        # ==================================================

        monotonic_features = []

        for feature in systematic_features:

            values = feature_values[feature]

            if self._is_monotonic(values):
                monotonic_features.append(feature)

        # ==================================================
        # Signal 3: decision transitions
        # ==================================================

        transitions = self._find_decision_transitions(
            decisions
        )

        decision_changes = len(transitions)

        # ==================================================
        # Signal 4: repeated queries
        # ==================================================

        task_keys = [
            tuple(
                query["task"][feature]
                for feature in FEATURES
            )
            for query in queries
        ]

        repeated_queries = (
            len(task_keys) - len(set(task_keys))
        )

        # ==================================================
        # Signal 5: query volume
        # ==================================================

        query_count = len(queries)

        # ==================================================
        # Boundary exploration
        # ==================================================

        boundary_exploration = []

        if transitions:

            for feature in monotonic_features:

                boundary_exploration.append(feature)

        # ==================================================
        # Store signals
        # ==================================================

        signals = {
            "systematic_features": systematic_features,
            "monotonic_features": monotonic_features,
            "boundary_exploration_features": boundary_exploration,
            "decision_changes": decision_changes,
            "decision_transition_positions": transitions,
            "repeated_queries": repeated_queries,
            "query_count": query_count,
        }

        # ==================================================
        # Score
        # ==================================================

        score = 0.0

        # Systematic variation
        if systematic_features:
            score += 0.30

        # Monotonic scanning
        if monotonic_features:
            score += 0.20

        # Decision transition
        if decision_changes >= self.decision_change_threshold:
            score += 0.20

        # Explicit boundary exploration
        if boundary_exploration:
            score += 0.20

        # Repeated queries
        if repeated_queries > 0:
            score += 0.05

        # High query volume
        if query_count >= self.window_size:
            score += 0.05

        score = min(score, 1.0)

        return {
            "probe_score": round(score, 3),
            "is_probe": score >= self.probe_threshold,
            "signals": signals,
        }