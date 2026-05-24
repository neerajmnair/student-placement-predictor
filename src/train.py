from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
):

    results = {}

    # Logistic Regression
    lr = LogisticRegression()

    lr.fit(
        X_train_scaled,
        y_train
    )

    lr_pred = lr.predict(
        X_test_scaled
    )

    results["Logistic Regression"] = (
        accuracy_score(
            y_test,
            lr_pred
        )
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        random_state=42
    )

    dt.fit(
        X_train,
        y_train
    )

    dt_pred = dt.predict(
        X_test
    )

    results["Decision Tree"] = (
        accuracy_score(
            y_test,
            dt_pred
        )
    )

    # Random Forest
    rf = RandomForestClassifier(
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    results["Random Forest"] = (
        accuracy_score(
            y_test,
            rf_pred
        )
    )

    # KNN
    knn = KNeighborsClassifier()

    knn.fit(
        X_train_scaled,
        y_train
    )

    knn_pred = knn.predict(
        X_test_scaled
    )

    results["KNN"] = (
        accuracy_score(
            y_test,
            knn_pred
        )
    )

    return results