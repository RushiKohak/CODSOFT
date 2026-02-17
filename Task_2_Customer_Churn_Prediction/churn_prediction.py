# CUSTOMER CHURN PREDICTION
# Beginner-friendly code for Codesoft Internship

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Step 2: Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

print("Dataset loaded successfully")
print(data.head())


# Step 3: Remove unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)


# Step 4: Encode categorical variables
label_encoder = LabelEncoder()

data["Geography"] = label_encoder.fit_transform(data["Geography"])
data["Gender"] = label_encoder.fit_transform(data["Gender"])


# Step 5: Separate features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]


# Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Step 7: Logistic Regression
# -----------------------------

# Scale the data (important for Logistic Regression)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)

# Predictions
lr_pred = lr.predict(X_test_scaled)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))


# -----------------------------
# Step 8: Random Forest
# -----------------------------

# Random Forest does NOT need scaling
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))
