import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import ( LSTM,Dense)

from evaluate import evaluate_model

# LOAD DATA

df = pd.read_csv(
    'data/processed/final_features.csv'
)


# SELECT STATE


state_name = 'California'

state_df = df[
    df['State'] == state_name
]


# SALES DATA


sales = state_df[
    'Sales'
].values.reshape(-1, 1)

# SCALE DATA


scaler = MinMaxScaler()

scaled_sales = scaler.fit_transform(
    sales
)

# CREATE SEQUENCES
X = []

y = []

sequence_length = 10

for i in range(
    sequence_length,
    len(scaled_sales)
):

    X.append(
        scaled_sales[
            i-sequence_length:i
        ]
    )

    y.append(
        scaled_sales[i]
    )

X = np.array(X)

y = np.array(y)

# TRAIN TEST SPLIT


split_index = int(
    len(X) * 0.8
)

X_train = X[:split_index]

X_valid = X[split_index:]

y_train = y[:split_index]

y_valid = y[split_index:]

# BUILD MODEL

model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(
            X_train.shape[1],
            1
        )
    )
)

model.add(
    Dense(1)
)

# COMPILE MODEL


model.compile(
    optimizer='adam',
    loss='mse'
)

# TRAIN MODEL

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16
)

# PREDICT

predictions = model.predict(
    X_valid
)

# INVERSE TRANSFORM

predictions = scaler.inverse_transform(
    predictions
)

y_valid = scaler.inverse_transform(
    y_valid
)

# EVALUATE

metrics = evaluate_model(
    y_valid,
    predictions
)

print(metrics)
model.save(
    'models/best_lstm_model.keras'
)

print(
    "Model saved successfully"
)