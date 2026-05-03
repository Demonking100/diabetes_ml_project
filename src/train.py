import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from data_preprocessing import load_data, clean_data, scale_features
from model import train_knn, evaluate

print("Training started...")

# Load dataset (IMPORTANT PATH)
df = load_data("../data/diabetes.csv")

print("Data loaded successfully")

# Clean
df = clean_data(df)

# Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale
X_scaled, scaler = scale_features(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

best_k = 1
best_acc = 0

# Train loop
best_k = 1
best_acc = 0

for k in range(3, 21, 2):   # only odd values: 3,5,7...
    model = train_knn(X_train, y_train, k)
    acc = evaluate(model, X_test, y_test)

    print(f"K={k}, Accuracy={acc}")  # debug ke liye

    if acc > best_acc:
        best_acc = acc
        best_k = k

print("Best K:", best_k)
print("Accuracy:", best_acc)

# Final model
final_model = train_knn(X_train, y_train, best_k)

# Save model
os.makedirs("../models", exist_ok=True)

joblib.dump(final_model, "../models/knn_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model saved successfully")