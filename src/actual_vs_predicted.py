import pandas as pd

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    'data/processed/final_features.csv'
)

# =========================
# SELECT STATE
# =========================

state_name = 'California'

state_df = df[
    df['State'] == state_name
]

# =========================
# FEATURES
# =========================

features = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'day_of_week',
    'month',
    'week',
    'year',
    'is_holiday'
]

X = state_df[features]

y = state_df['Sales']

# =========================
# SPLIT
# =========================

split_index = int(
    len(state_df) * 0.8
)

X_train = X.iloc[:split_index]

X_valid = X.iloc[split_index:]

y_train = y.iloc[:split_index]

y_valid = y.iloc[split_index:]

# =========================
# TRAIN MODEL
# =========================

model = XGBRegressor()

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICT
# =========================

predictions = model.predict(
    X_valid
)

# =========================
# PLOT
# =========================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    y_valid.values,
    label='Actual'
)

plt.plot(
    predictions,
    label='Predicted'
)

plt.title(
    'Actual vs Predicted Sales'
)

plt.xlabel(
    'Time'
)

plt.ylabel(
    'Sales'
)

plt.legend()

plt.show()