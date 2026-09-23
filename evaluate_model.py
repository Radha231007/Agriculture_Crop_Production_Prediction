import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
project_folder = Path(__file__).resolve().parent.parent
processed_folder = project_folder / "data" / "processed"
models_folder = project_folder / "models"
figures_folder = project_folder / "results" / "figures"

figures_folder.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(processed_folder / "agriculture_cleaned.csv")

# Encode crop
encoder = LabelEncoder()
df["Crop_encoded"] = encoder.fit_transform(df["Crop"])

# Convert year
df["Year_numeric"] = df["Year"].str[:4].astype(int)

# Features and target
X = df[["Crop_encoded", "Year_numeric"]]
y = df["Production"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load(
    models_folder / "agriculture_production_random_forest.pkl"
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Agriculture Production Prediction")
print("=" * 45)
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))

# Actual vs predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Production")
plt.ylabel("Predicted Production")
plt.title("Actual vs Predicted Agriculture Production")
plt.tight_layout()

graph_file = figures_folder / "actual_vs_predicted.png"
plt.savefig(graph_file)
plt.close()

print("\nGraph saved to:")
print(graph_file)