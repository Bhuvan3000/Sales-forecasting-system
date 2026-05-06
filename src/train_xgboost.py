import pandas as pd

from xgboost import XGBRegressor

from evaluate import evaluate_model

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
# TRAIN TEST SPLIT
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

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

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
# EVALUATE
# =========================

metrics = evaluate_model(
    y_valid,
    predictions
)

print(metrics)