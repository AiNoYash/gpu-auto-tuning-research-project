import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error


df = pd.read_csv('gemm_tuning_dataset_1.csv')

# Handle the -1.0 latency condition (replacing with 10^6)
df.loc[df['latency_ms'] == -1.0, 'latency_ms'] = 1e6

# Drop the columns we won't use
columns_to_drop = ['status', 'mean_ms', 'variance', 'rel_variance', 'iterations']
df = df.drop(columns=columns_to_drop)

# Separate features (X) and target (y)
X = df.drop(columns=['latency_ms'])
y = df['latency_ms']

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")




# Initialize the XGBoost regressor
# 'reg:squaredlogerror' minimizes the Root Mean Squared Logarithmic Error (RMSLE)
model = xgb.XGBRegressor(
    objective='reg:squaredlogerror',
    n_estimators=100,      # Number of trees
    learning_rate=0.1,     # Step size (how aggressively it learns)
    max_depth=6,           # Maximum depth of a tree
    random_state=42
)

print("Training the XGBoost model...")
# Train the model
model.fit(X_train, y_train)
print("Training complete!")

# Generate predictions on the test set
predictions = model.predict(X_test)

# Display a few predictions vs actual values to see how it did
print("\nSample Predictions vs Actual Latency:")
for i in range(5):
    print(f"Predicted: {predictions[i]:.2f} ms | Actual: {y_test.iloc[i]:.2f} ms")
    


# Ensure no negative predictions before calculating log error
# (XGBoost can occasionally predict slightly below 0, which breaks log calculations)
predictions = np.clip(predictions, 0, None)

# Calculate metrics
rmsle = np.sqrt(mean_squared_log_error(y_test, predictions))
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("\nModel Evaluation Metrics:")
print(f"RMSLE (Relative Error focus): {rmsle:.4f}")
print(f"RMSE (Absolute Error focus):  {rmse:.4f}")
print(f"MAE (Mean Absolute Error):    {mae:.4f}")