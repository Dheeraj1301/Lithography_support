import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_processed_data():
    df = pd.read_csv("data/raw/synthetic_litho.csv")
    features = df.drop("critical_dimension", axis=1)
    target = df["critical_dimension"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
    return X_train, y_train
