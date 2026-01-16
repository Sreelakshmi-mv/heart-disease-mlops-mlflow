import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

# Set MLflow experiment
mlflow.set_experiment("Heart-Disease-EDA")

with mlflow.start_run(run_name="EDA_Run"):
    df = load_data()

    # Log basic dataset info
    mlflow.log_param("rows", df.shape[0])
    mlflow.log_param("columns", df.shape[1])

    # Target distribution
    plt.figure()
    sns.countplot(x="target", data=df)
    plt.title("Target Distribution")
    plt.savefig("target_distribution.png")
    mlflow.log_artifact("target_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    mlflow.log_artifact("correlation_heatmap.png")
    plt.close()

    print("EDA completed and logged to MLflow")
