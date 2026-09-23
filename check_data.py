import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
raw_folder = project_folder / "data" / "raw"

csv_files = list(raw_folder.glob("*.csv"))

print("CSV files found:", len(csv_files))
print("=" * 60)

for file in csv_files:
    print("\nFile:", file.name)

    try:
        df = pd.read_csv(file)

        print("Shape:", df.shape)
        print("Columns:")
        print(df.columns.tolist())

        print("\nFirst 3 rows:")
        print(df.head(3))

    except Exception as e:
        print("Error reading file:", e)

    print("=" * 60)