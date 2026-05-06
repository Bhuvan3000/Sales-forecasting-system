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

- Lag Features
  - t-1
  - t-7
  - t-30

- Rolling Features
  - Rolling Mean
  - Rolling Standard Deviation

- Date Features
  - Day of week
  - Month
  - Week
  - Year

- Holiday Flag

---

## Evaluation Metrics

- MAE
- RMSE
- MAPE

---

## Model Performance

| Model | RMSE |
|---|---|
| SARIMA | 109885876 |
| Prophet | 179685006 |
| XGBoost | 77168688 |
| LSTM | 48124568 |

Best Model: LSTM

---

## API Endpoint

```bash
GET /forecast/{state}
```

Example:

```bash
/forecast/California
```

---

## Run Project

```bash
uvicorn api.main:app --reload
```

---

## Swagger Docs

```bash
http://127.0.0.1:8000/docs
```

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TensorFlow
- Prophet
- XGBoost
- FastAPI

---

## Future Improvements

- Real-time forecasting
- Cloud deployment
- Docker support
- Streamlit dashboard