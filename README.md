# Sales Forecasting System

## Project Overview

This project demonstrates a production-ready time series forecasting pipeline using machine learning and deep learning models with REST API deployment.

The system forecasts the next 8 weeks of sales for each state using historical sales data.

---

## Features

- Data preprocessing
- Missing value handling
- Feature engineering
- Multiple forecasting models
- Automatic model comparison
- REST API using FastAPI
- Swagger API documentation

---

## Models Implemented

1. SARIMA
2. Facebook Prophet
3. XGBoost
4. LSTM

---

## Feature Engineering

The following features were created:

### Lag Features
- t-1
- t-7
- t-30

### Rolling Features
- Rolling Mean
- Rolling Standard Deviation

### Date Features
- Day of week
- Month
- Week
- Year

### Holiday Feature
- Indian Holiday Flag

---

## Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

## Model Performance

| Model | RMSE |
|---|---|
| SARIMA | 109885876 |
| Prophet | 179685006 |
| XGBoost | 77168688 |
| LSTM | 48124568 |

### Best Model
LSTM achieved the lowest RMSE and highest forecasting accuracy.

---

## Visualization

The project includes:

- Model comparison graph
- Actual vs Predicted visualization
- Forecast output visualization

---

## API Endpoint

### Forecast Endpoint

```bash
GET /forecast/{state}
```

### Example

```bash
/forecast/California
```

### Example Response

```json
{
  "state": "California",
  "forecast_next_8_weeks": [
    48124568,
    49200000,
    50300000,
    51000000,
    52200000,
    53000000,
    54000000,
    55200000
  ]
}
```

---

## Run Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI Server

```bash
uvicorn api.main:app --reload
```

---

## Swagger API Documentation

Open browser:

```bash
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
sales_forecasting_system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_arima.py
│   ├── train_prophet.py
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── model_comparison.py
│   ├── visualize.py
│   └── actual_vs_predicted.py
│
├── api/
│   └── main.py
│
├── models/
│   └── best_lstm_model.keras
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- TensorFlow / Keras
- FastAPI
- Matplotlib

---

## Future Improvements

- Real-time forecasting
- Docker deployment
- Cloud deployment
- Streamlit dashboard
- Automated retraining pipeline

---

## Conclusion

This project successfully implements an end-to-end time series forecasting system with machine learning, deep learning, model comparison, and REST API deployment using FastAPI.