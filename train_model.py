import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folders
project_folder = Path(__file__).resolve().parent.parent
processed_folder = project_folder / "data" / "processed"
models_folder = project_folder / "models"

models_folder.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
file_path = processed_folder / "agriculture_cleaned.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Encode Crop
crop_encoder = LabelEncoder()
df["Crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

# Convert Year to starting year
df["Year_numeric"] = df["Year"].str[:4].astype(int)

# Features and target
X = df[["Crop_encoded", "Year_numeric"]]
y = df["Production"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Results")
print("-" * 40)
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))

# Save model
model_file = models_folder / "agriculture_production_random_forest.pkl"

import joblib
joblib.dump(model, model_file)

print("\nModel saved to:")
print(model_file)