from .lore_explainer import LOREExplainer
from .query_logger import QueryLogger
from .explanation_guard import ExplanationGuard


class ExplanationService:
    """
    End-to-end service connecting:

        Scheduler
            ↓
        Query Logger
            ↓
        Probe Detector / Guard
            ↓
        LOREsa

    The scheduler always produces a decision.
    The guard controls whether the explanation is released.
    """

    def __init__(
        self,
        scheduler,
        reference_data,
        num_instances=500,
    ):
        self.scheduler = scheduler

        self.explainer = LOREExplainer(
            scheduler=scheduler,
            reference_data=reference_data,
            num_instances=num_instances,
        )

        self.logger = QueryLogger()
        self.guard = ExplanationGuard()

    def process_task(
        self,
        task,
        explanation_requested=True,
    ):
        """
        Process one scheduling request.

        Returns the scheduler decision and, if allowed,
        the LOREsa explanation.
        """

        # --------------------------------------------------
        # 1. Scheduler makes its decision
        # --------------------------------------------------

        decision = self.scheduler.predict(task)[0]

        # --------------------------------------------------
        # 2. Record the query
        # --------------------------------------------------

        self.logger.log_query(
            task=task,
            decision=decision,
            explanation_requested=explanation_requested,
            explanation_released=False,
        )

        # --------------------------------------------------
        # 3. Evaluate current query behavior
        # --------------------------------------------------

        guard_result = self.guard.evaluate(
            self.logger.get_history()
        )

        # --------------------------------------------------
        # 4. Release explanation only if allowed
        # --------------------------------------------------

        explanation = None

        if (
            explanation_requested
            and guard_result["explanation_allowed"]
        ):

            explanation = self.explainer.explain(task)

            # Update the most recent log entry
            self.logger.history[-1]["explanation_released"] = True

        return {
            "decision": decision,
            "explanation": explanation,
            "probe_score": guard_result["probe_score"],
            "is_probe": guard_result["is_probe"],
            "explanation_allowed": (
                guard_result["explanation_allowed"]
            ),
            "signals": guard_result["signals"],
        }