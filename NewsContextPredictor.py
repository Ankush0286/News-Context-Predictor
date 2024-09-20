# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset with error handling
data = pd.read_csv(r"D:\Project Datasets\Fake.csv")

# Fill any missing values with an empty string
data.fillna('', inplace=True)

# Combine title, text, and subject into a single feature if subject is useful
data['content'] = data['title'] + ' ' + data['text'] + ' ' + data['subject']

# Define features (X) and target (y)
X = data['content']
y = data['subject']  # Adjust this line if you have a different target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print confusion matrix and classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Example user input and prediction
def predict_subject(news_text):
    input_data = [news_text]
    input_tfidf = tfidf_vectorizer.transform(input_data)
    prediction = model.predict(input_tfidf)
    return prediction[0]

# Test the prediction
news_text = "NASA will send people to moon soon"
print(f"\nPrediction: {predict_subject(news_text)}")
