import pandas as pd

# Load Excel file
file_path = "data/raw/Forecasting Case- Study.xlsx"

# Read dataset
df = pd.read_excel(file_path)

# Show first rows
print(df.head())

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort values
df = df.sort_values(['State', 'Date'])

# Rename Total -> Sales
# Easier for forecasting

df = df.rename(columns={'Total': 'Sales'})

# Check missing values
print(df.isnull().sum())

# Fill missing sales values

df['Sales'] = df['Sales'].interpolate()

# Save cleaned data

df.to_csv('data/processed/cleaned_sales.csv', index=False)

print("Data preprocessing completed")