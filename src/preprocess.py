import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary column
    df = df.drop("StudentID", axis=1)

    # Encode categorical variables
    le = LabelEncoder()

    columns = df.select_dtypes(
        include="str"
    ).columns

    for col in columns:
        df[col] = le.fit_transform(
            df[col]
        )

    # Features and target
    X = df.drop(
        "PlacementStatus",
        axis=1
    )

    y = df["PlacementStatus"]

    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    )