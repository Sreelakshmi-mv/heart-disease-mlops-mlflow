import pandas as pd

# UCI Heart Disease Dataset (Cleveland)
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

def load_data():
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS)

    # Handle missing values
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Convert target to binary
    df["target"] = df["target"].astype(int).apply(lambda x: 1 if x > 0 else 0)

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nShape:", df.shape)
