# Goal: Load data, explore it, and handle missing values.
import numpy as np
from sklearn.datasets import fetch_california_housing
print("=" * 60)
print("PHASE 1: LOADING & EXPLORING THE DATA")
print("=" * 60)
# 1. Load the dataset 
housing = fetch_california_housing()
x = housing.data        # Features (the inputs)
y = housing.target      # Target (the house prices)
print("Data Loaded Successfully!")
print(f"Total samples: {x.shape[0]}")
print(f"Total features: {x.shape[1]}")
# Columns in the dataset!
print("\n Features in this dataset:")
for i, name in enumerate(housing.feature_names):
    print(f"{i+1}. {name}")
# Checking for Missing data
print("\nChecking for Missing values...")
nan_count = np.sum(np.isnan(x))
if nan_count == 0:
    print("No missing values!")
else:
    print(f"Found {nan_count} missing values!")
print("\nQuick stats")
print(f" Average Price: ${np.mean(y) * 100000:.2f}")
print(f"Minimum Price: ${np.min(y) * 100000:.2f}")
print(f"Maximum Price: ${np.max(y) * 100000:.2f}")
print("\n" + "=" * 60)
print("🧠 PHASE 2: FEATURE ENGINEERING")
print("=" * 60)
rooms_per_household = x[:,2] / x[:,5]
x = np.column_stack((x, rooms_per_household))
# print(rooms_per_household)
# avgrooms = x[:, 2]
# print(avgrooms)
# avgoccup = x[:, 5]
# print(avgoccup)
print(f"New column added, Total columns: {x.shape}")
print("\n" + "=" * 60)
print("⚙️ PHASE 3: SPLITTING & NORMALIZING")
print("=" * 60)
np.random.seed(42) # For reproducibilty
indicies = np.random.permutation(len(x))
x = x[indicies]
y = y[indicies]
# Split: 80% Train, 20% Test
split_idx = int(0.8 * len(x))
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print("Data split complete:")
print(f"Training set: {len(x_train)} samples")
print(f"Testing set: {len(x_test)} samples")
# 3. Normalize the features (Z-score)
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std
print("Features normalized!")
print("\n" + "=" * 60)
print("PHASE 4: BUILDING THE MODEL (Normal Equation)")
print("=" * 60)
# 1. Add a bias term (a column of 1s) to the training data
x_train_bias = np.hstack([np.ones((x_train_norm.shape[0], 1)), x_train_norm])
# 2. Calculate the weights using the Normal Equation
# Formula: weights = (X^T X)^-1 X^T y
weights = np.linalg.inv(x_train_bias.T @ x_train_bias) @ x_train_bias.T @ y_train
# 3. Make predictions on the TEST set
x_test_bias = np.hstack([np.ones((x_test_norm.shape[0], 1)), x_test_norm])
y_pred = x_test_bias @ weights
# 4. Evaluate performance 
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
print("\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): ${rmse * 100000:.2f}")
# 5. Show Feature Importance
print("\n Feature Importance (Weight values):")
feature_names_with_bias = ["Bias"] + list(housing.feature_names) + ["RoomsPerHH"]
for i, weight in enumerate(weights):
    if i == 0:
        print(f"{feature_names_with_bias[i]}: {weight:.4f} (Base vale)")
    else:
        print(f"{feature_names_with_bias}: {weight:.4f}")