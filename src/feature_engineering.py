import pandas as pd
import holidays

# Load cleaned data

df = pd.read_csv('data/processed/cleaned_sales.csv')

# Convert date again

df['Date'] = pd.to_datetime(df['Date'])

# India holidays
india_holidays = holidays.India()

# Create features

# Lag features

df['lag_1'] = df.groupby('State')['Sales'].shift(1)
df['lag_7'] = df.groupby('State')['Sales'].shift(7)
df['lag_30'] = df.groupby('State')['Sales'].shift(30)

# Rolling statistics

df['rolling_mean_7'] = (
    df.groupby('State')['Sales']
    .transform(lambda x: x.rolling(7).mean())
)

# Rolling std

df['rolling_std_7'] = (
    df.groupby('State')['Sales']
    .transform(lambda x: x.rolling(7).std())
)

# Date features

df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['week'] = df['Date'].dt.isocalendar().week.astype(int)
df['year'] = df['Date'].dt.year

# Holiday flag

df['is_holiday'] = df['Date'].apply(
    lambda x: 1 if x in india_holidays else 0
)

# Remove rows with NaN created by lagging

df = df.dropna()

# Save feature dataset

df.to_csv('data/processed/final_features.csv', index=False)

print("Feature engineering completed")