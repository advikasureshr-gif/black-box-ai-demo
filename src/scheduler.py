import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from .scheduler_data import (
    generate_scheduling_decisions,
    FEATURES,
)


class TaskScheduler:
    """
    Black-box ML task scheduler.

    The underlying scheduling policy is used only to generate
    training labels. Once trained, the Random Forest acts as
    the black-box decision-maker exposed to LOREsa.
    """

    def __init__(
        self,
        n_estimators=100,
        random_state=42,
    ):
        self.vm_names = [
            "VM-1",
            "VM-2",
            "VM-3",
            "VM-4",
        ]

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=8,
            min_samples_leaf=3,
        )

        self._train()

    def _train(self):

        training_data = self._generate_training_data()

        X = training_data[FEATURES]
        y = training_data["assigned_vm"]

        self.model.fit(X, y)

    def _generate_training_data(self):

        rng = np.random.default_rng(42)

        data = pd.DataFrame({
            "cpu_required": rng.integers(1, 17, 5000),
            "memory_required": rng.integers(1, 33, 5000),
            "priority": rng.integers(1, 11, 5000),
            "deadline": rng.integers(20, 301, 5000),
            "queue_length": rng.integers(0, 21, 5000),
        })

        data["assigned_vm"] = generate_scheduling_decisions(
            data
        )

        return data

    def _prepare_input(self, X):

        if isinstance(X, pd.DataFrame):

            dataframe = X.copy()

        elif isinstance(X, np.ndarray):

            if X.ndim == 1:
                X = X.reshape(1, -1)

            dataframe = pd.DataFrame(
                X,
                columns=FEATURES,
            )

        elif isinstance(X, list):

            dataframe = pd.DataFrame(
                X,
                columns=FEATURES,
            )
            
        elif isinstance(X, dict):
            dataframe = pd.DataFrame([X])

        else:

            raise TypeError(
                f"Unsupported input type: {type(X)}"
            )

        return dataframe[FEATURES]

    def predict(self, X):

        dataframe = self._prepare_input(X)

        return self.model.predict(dataframe)

    def predict_proba(self, X):

        dataframe = self._prepare_input(X)

        return self.model.predict_proba(dataframe)