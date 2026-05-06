import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from evaluate import evaluate_model

# Load data

df = pd.read_csv('data/processed/final_features.csv')

# Choose one state first
state_name = 'California'

state_df = df[df['State'] == state_name]

# Train-validation split
split_index = int(len(state_df) * 0.8)

train = state_df.iloc[:split_index]
valid = state_df.iloc[split_index:]

# Train SARIMA

model = SARIMAX(
    train['Sales'],
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

results = model.fit(disp=False)

# Predict

predictions = results.forecast(steps=len(valid))

# Evaluate

metrics = evaluate_model(valid['Sales'], predictions)

print(metrics)