from fastapi import FastAPI

app = FastAPI()

# HOME ROUTE


@app.get('/')

def home():

    return {
        'message': 'Sales Forecast API Running'
    }

# FORECAST ROUTE

@app.get('/forecast/{state}')

def forecast(state: str):

    # Dummy predictions

    predictions = [
    48124568,
    49200000,
    50300000,
    51000000,
    52200000,
    53000000,
    54000000,
    55200000
]

    return {
        'state': state,
        'forecast_next_8_weeks': predictions
    }