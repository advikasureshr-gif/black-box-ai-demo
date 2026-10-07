from datetime import datetime
import pandas as pd


class QueryLogger:
    """
    Records task-scheduling queries and the resulting decisions.

    The logger does not decide whether a user is probing.
    It simply maintains the query history that the probe detector
    will analyze later.
    """

    def __init__(self):
        self.history = []

    def log_query(
        self,
        task,
        decision,
        explanation_requested=False,
        explanation_released=False,
        probe_score=None,
    ):
        """
        Record one scheduler interaction.
        """

        if isinstance(task, pd.Series):
            task = task.to_dict()
        elif isinstance(task, pd.DataFrame):
            task = task.iloc[0].to_dict()

        record = {
            "timestamp": datetime.now().isoformat(),
            "task": dict(task),
            "decision": decision,
            "explanation_requested": explanation_requested,
            "explanation_released": explanation_released,
            "probe_score": probe_score,
        }

        self.history.append(record)

        return record

    def get_history(self):
        """Return all recorded queries."""
        return self.history

    def get_recent(self, n=10):
        """Return the most recent n queries."""
        return self.history[-n:]

    def clear(self):
        """Clear the current query history."""
        self.history.clear()

    def __len__(self):
        return len(self.history)