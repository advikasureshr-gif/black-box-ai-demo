
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_models(df):
    # Fair model: age is not an input feature
    X_fair = df[["proficiency"]]
    y_fair = df["fair_hired"]

    fair_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    fair_model.fit(X_fair, y_fair)

    # Biased model: age and proficiency are both input features
    X_biased = df[["age", "proficiency"]]
    y_biased = df["biased_hired"]

    biased_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    biased_model.fit(X_biased, y_biased)

    return fair_model, biased_model
