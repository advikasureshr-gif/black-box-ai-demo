import pandas as pd

from lore_sa.lore import TabularGeneticGeneratorLore
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearnBBox

from .scheduler_data import FEATURES


class LOREExplainer:
    """
    Adapter between our black-box task scheduler and LOREsa.

    LOREsa itself is kept untouched. This class handles:
        - dataset preparation
        - black-box wrapping
        - LOREsa initialization
        - local explanation generation
    """

    def __init__(
        self,
        scheduler,
        reference_data,
        num_instances=500,
    ):
        self.scheduler = scheduler
        self.reference_data = reference_data
        self.num_instances = num_instances

        # ------------------------------------------
        # Wrap the scheduler as a LOREsa black box
        # ------------------------------------------

        self.bbox = sklearnBBox(
            scheduler
        )

        # ------------------------------------------
        # Tell LOREsa about our tabular dataset
        # ------------------------------------------

        self.dataset = TabularDataset(
            reference_data,
            class_name="assigned_vm",
        )

        # ------------------------------------------
        # Create the LOREsa explainer
        # ------------------------------------------

        self.explainer = TabularGeneticGeneratorLore(
            self.bbox,
            self.dataset,
        )

    def explain(self, task):
        """
        Generate a LOREsa explanation for one task.

        Parameters
        ----------
        task : dict, pandas.Series, or array-like
            Task features.

        Returns
        -------
        dict
            LOREsa explanation containing:
                rule
                counterfactuals
                fidelity
                feature_importances
        """

        # ------------------------------------------
        # Convert task to a DataFrame
        # ------------------------------------------

        if isinstance(task, dict):

            instance = pd.DataFrame(
                [task],
                columns=FEATURES,
            )

        elif isinstance(task, pd.Series):

            instance = pd.DataFrame(
                [task[FEATURES].to_dict()],
                columns=FEATURES,
            )

        elif isinstance(task, pd.DataFrame):

            instance = task[FEATURES].iloc[
                [0]
            ].copy()

        else:

            instance = pd.DataFrame(
                [task],
                columns=FEATURES,
            )

        # ------------------------------------------
        # Convert to NumPy array for LOREsa
        # ------------------------------------------

        x = instance.iloc[0].values

        # ------------------------------------------
        # Get black-box prediction
        # ------------------------------------------

        prediction = self.scheduler.predict(
            instance
        )[0]

        # ------------------------------------------
        # Generate LOREsa explanation
        # ------------------------------------------

        explanation = self.explainer.explain(
            x,
            num_instances=self.num_instances,
        )

        # ------------------------------------------
        # Return a clean application-level result
        # ------------------------------------------

        return {
            "prediction": prediction,
            "rule": explanation["rule"],
            "counterfactuals": explanation[
                "counterfactuals"
            ],
            "fidelity": explanation[
                "fidelity"
            ],
            "feature_importances": explanation[
                "feature_importances"
            ],
        }