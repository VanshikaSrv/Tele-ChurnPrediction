import joblib
import pandas as pd
import numpy as np


def load_model():

    model = joblib.load(
        "models/best_model.pkl"
    )

    return model


def get_top_reasons(customer_data, top_n=5):

    model = load_model()

    if not hasattr(model, "named_steps"):
        raise Exception(
            "Saved model is not a valid sklearn Pipeline."
        )

    preprocessor = model.named_steps["preprocessor"]

    ml_model = model.named_steps["model"]

    input_df = pd.DataFrame(
        [customer_data]
    )

    processed_input = (
        preprocessor.transform(
            input_df
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    if hasattr(processed_input, "toarray"):

        values = processed_input.toarray()[0]

    else:

        values = processed_input[0]

    # Logistic Regression
    if hasattr(ml_model, "coef_"):

        coefficients = ml_model.coef_[0]

        impacts = coefficients * values

    # Decision Tree / Random Forest
    elif hasattr(
        ml_model,
        "feature_importances_"
    ):

        importances = (
            ml_model.feature_importances_
        )

        impacts = importances * values

    else:

        raise Exception(
            "Current model does not support "
            "feature explanation."
        )

    explanation_df = pd.DataFrame({

        "Feature": feature_names,

        "Value": values,

        "Impact": impacts

    })

    explanation_df["AbsImpact"] = (
        explanation_df["Impact"].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "AbsImpact",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    explanation_df["Direction"] = np.where(
        explanation_df["Impact"] > 0,
        "Increases Churn",
        "Decreases Churn"
    )

    return explanation_df


def clean_feature_name(feature):

    clean_feature = (
        str(feature)
        .replace("cat__", "")
        .replace("num__", "")
        .replace("onehot__", "")
        .replace("_", " ")
    )

    return clean_feature


def generate_explanation(top_features):

    reasons = []

    for _, row in top_features.iterrows():

        feature = clean_feature_name(
            row["Feature"]
        )

        direction = row["Direction"]

        if direction == "Increases Churn":

            reasons.append(
                feature
                + " increases the likelihood of churn"
            )

        else:

            reasons.append(
                feature
                + " decreases the likelihood of churn"
            )

    if not reasons:

        return "No major influencing factors were found."

    explanation = (
        "The prediction is mainly influenced by:\n\n"
        + "\n".join(
            "• " + reason
            for reason in reasons
        )
    )

    return explanation