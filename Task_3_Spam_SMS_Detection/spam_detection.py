# SPAM SMS DETECTION
# Beginner-friendly code for Codesoft Internship

# Step 1: Import required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Step 2: Load the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

print("Dataset loaded successfully")
print(data.head())


# Step 3: Keep only useful columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]


# Step 4: Convert labels to numeric form
# ham = 0, spam = 1
data["label"] = data["label"].map({"ham": 0, "spam": 1})


# Step 5: Separate features and target
X = data["message"]
y = data["label"]


# Step 6: Convert text messages into numerical data using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
X_tfidf = tfidf.fit_transform(X)


# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)


# Step 8: Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))


# Step 9: Logistic Regression Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
