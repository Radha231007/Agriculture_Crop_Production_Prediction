import pandas as pd
from pathlib import Path

# Project folders
project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"
processed_folder = project_folder / "data" / "processed"

# Create processed folder if it doesn't exist
processed_folder.mkdir(parents=True, exist_ok=True)

# Read the UCT production dataset
file_path = raw_folder / "datafile (2).csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# First column is the crop name
crop_column = df.columns[0]

# Production columns
production_columns = [
    "Production 2006-07",
    "Production 2007-08",
    "Production 2008-09",
    "Production 2009-10",
    "Production 2010-11"
]

# Convert from wide format to long format
cleaned_df = df[[crop_column] + production_columns].melt(
    id_vars=[crop_column],
    var_name="Year",
    value_name="Production"
)

# Rename crop column
cleaned_df = cleaned_df.rename(columns={crop_column: "Crop"})

# Extract year
cleaned_df["Year"] = cleaned_df["Year"].str.replace("Production ", "")

# Remove missing values
cleaned_df = cleaned_df.dropna()

# Save cleaned dataset
output_file = processed_folder / "agriculture_cleaned.csv"
cleaned_df.to_csv(output_file, index=False)

print("Data preparation completed!")
print("Cleaned dataset shape:", cleaned_df.shape)
print("\nColumns:")
print(cleaned_df.columns.tolist())
print("\nFirst 10 rows:")
print(cleaned_df.head(10))
print("\nSaved to:")
print(output_file)