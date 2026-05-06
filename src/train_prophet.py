import pandas as pd
from prophet import Prophet
from evaluate import evaluate_model

# Load data

df = pd.read_csv('data/processed/final_features.csv')

state_name = 'California'

state_df = df[df['State'] == state_name]

# Prophet format

prophet_df = state_df[['Date', 'Sales']]

prophet_df = prophet_df.rename(
    columns={
        'Date': 'ds',
        'Sales': 'y'
    }
)

# Split

split_index = int(len(prophet_df) * 0.8)

train = prophet_df.iloc[:split_index]
valid = prophet_df.iloc[split_index:]

# Train

model = Prophet()

model.fit(train)

# Future dataframe

future = model.make_future_dataframe(periods=len(valid))

forecast = model.predict(future)

predictions = forecast['yhat'].tail(len(valid))

# Evaluate

metrics = evaluate_model(valid['y'], predictions)

print(metrics)