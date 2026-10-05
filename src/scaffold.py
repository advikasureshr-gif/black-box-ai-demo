
import pandas as pd


class ScaffoldedRecruitmentModel:
    def __init__(self, fair_model, biased_model):
        self.fair_model = fair_model
        self.biased_model = biased_model

    def predict(self, profiles, is_probe=False):
        """
        Simulate a scaffolding attack.

        is_probe=True:
            Route explanation probes to the fair model.

        is_probe=False:
            Route ordinary candidate profiles to the biased model.
        """
        if is_probe:
            inputs = profiles[["proficiency"]]
            return self.fair_model.predict(inputs)

        inputs = profiles[["age", "proficiency"]]
        return self.biased_model.predict(inputs)

    def predict_proba(self, profiles, is_probe=False):
        """Return hiring probabilities from the selected model."""
        if is_probe:
            inputs = profiles[["proficiency"]]
            return self.fair_model.predict_proba(inputs)

        inputs = profiles[["age", "proficiency"]]
        return self.biased_model.predict_proba(inputs)
