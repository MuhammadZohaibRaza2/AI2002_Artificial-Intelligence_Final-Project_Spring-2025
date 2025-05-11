import pandas as pd

csv_path = 'telemetry_log.csv'  # Adjust path if needed
df = pd.read_csv(csv_path)
print("Columns in telemetry_log.csv:")
print(df.columns.tolist())
df.head()