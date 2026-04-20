import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

print("1. Imports successful! Loading data...")
# Load dataset
data = pd.read_csv("data/crop_production.csv")
# ... (imports stay the same)

print("1. Imports successful! Loading data...")
# Load dataset
data = pd.read_csv("data/crop_production.csv")

# ADD IT HERE (Right after loading)
print("Reducing dataset size for faster training...")
data = data.sample(n=5000, random_state=42) 

print("2. Data loaded. Cleaning and prepping features...")
# Clean data
data = data.dropna()

# ... (rest of your code stays the same)

print("2. Data loaded. Cleaning and prepping features...")
# Clean data
data = data.dropna()

# Separate target
y = data["Production"]

# Drop target from features
X = data.drop("Production", axis=1)

# Convert categorical columns into numbers
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"3. Training Random Forest on {X_train.shape[0]} rows and {X_train.shape[1]} features...")
print("This might take a few minutes depending on your CPU. Hang tight!")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("4. Training complete! Saving files...")

# Automatically create the 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save BOTH model and columns (VERY IMPORTANT)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

print("Best model trained and saved successfully!")