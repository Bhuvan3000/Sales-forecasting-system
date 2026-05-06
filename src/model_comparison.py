import pandas as pd

# =========================
# MODEL RESULTS
# =========================

results = {
    'Model': [
        'SARIMA',
        'Prophet',
        'XGBoost',
        'LSTM'
    ],

    'RMSE': [
        109885876,
        179685006,
        77168688,
        48124568
    ]
}

# =========================
# CREATE DATAFRAME
# =========================

results_df = pd.DataFrame(
    results
)

# =========================
# BEST MODEL
# =========================

best_model = results_df.loc[
    results_df['RMSE'].idxmin()
]

print()

print(
    "Best Model:"
)

print(best_model)

print()

print(
    "All Results:"
)

print(results_df)

# =========================
# SAVE CSV
# =========================

results_df.to_csv(
    'model_results.csv',
    index=False
)

print()

print(
    "Results saved to model_results.csv"
)