import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error

df = pd.read_csv('gemm_tuning_dataset_1.csv')
df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e6
columns_to_drop = ['status', 'mean_ms', 'variance', 'rel_variance', 'iterations']
df = df.drop(columns=columns_to_drop)

X = df.drop(columns=['latency_ms'])
y = df['latency_ms']

y_log = np.log1p(y)


X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
_, _, _, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model using standard squared error (since data is already logged)
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

print("Training the XGBoost model on log-transformed data...")

model.fit(X_train, y_train_log)
print("Training complete!")

# Generate predictions and reverse the log transformation
predictions_log = model.predict(X_test)
predictions = np.expm1(predictions_log) # Inverse of log1p

# Ensure no negative predictions before calculating log error
predictions = np.clip(predictions, 0, None)

# Evaluate and display
print("\nSample Predictions vs Actual Latency:")
for i in range(5):
    print(f"Predicted: {predictions[i]:.2f} ms | Actual: {y_test_orig.iloc[i]:.2f} ms")

rmsle = np.sqrt(mean_squared_log_error(y_test_orig, predictions))
rmse = np.sqrt(mean_squared_error(y_test_orig, predictions))
mae = mean_absolute_error(y_test_orig, predictions)

print("\nModel Evaluation Metrics:")
print(f"RMSLE (Relative Error focus): {rmsle:.4f}")
print(f"RMSE (Absolute Error focus):  {rmse:.4f}")
print(f"MAE (Mean Absolute Error):    {mae:.4f}")