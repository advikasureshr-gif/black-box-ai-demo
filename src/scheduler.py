import numpy as np
import pandas as pd

from .scheduler_data import (
    generate_scheduling_decisions,
    FEATURES,
)


class TaskScheduler:

    def __init__(self):
        self.vm_names = [
            "VM-1",
            "VM-2",
            "VM-3",
            "VM-4",
        ]

    def predict(self, X):

        if isinstance(X, pd.DataFrame):
            dataframe = X.copy()

        elif isinstance(X, np.ndarray):

            if X.ndim == 1:
                X = X.reshape(1, -1)

            dataframe = pd.DataFrame(
                X,
                columns=FEATURES
            )

        elif isinstance(X, list):

            dataframe = pd.DataFrame(
                X,
                columns=FEATURES
            )

        else:
            raise TypeError(
                f"Unsupported input type: {type(X)}"
            )

        return generate_scheduling_decisions(dataframe)

    def predict_proba(self, X):

        predictions = self.predict(X)

        probabilities = np.zeros(
            (len(predictions), len(self.vm_names))
        )

        for i, prediction in enumerate(predictions):

            vm_index = self.vm_names.index(
                prediction
            )

            probabilities[i, vm_index] = 1.0

        return probabilities