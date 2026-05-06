import matplotlib.pyplot as plt

models = [
    'SARIMA',
    'Prophet',
    'XGBoost',
    'LSTM'
]

rmse_scores = [
    109885876,
    179685006,
    77168688,
    48124568
]

plt.figure(
    figsize=(10, 5)
)

plt.bar(
    models,
    rmse_scores
)

plt.title(
    'Model Comparison'
)

plt.xlabel(
    'Models'
)

plt.ylabel(
    'RMSE'
)

plt.show()