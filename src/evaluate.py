from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


def evaluate_model(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    return cm, report