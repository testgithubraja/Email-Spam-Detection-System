# Email-Spam-Detection-System
#*This project is a smart Email Spam Management System that leverages Machine Learning (ML), Deep Learning (DL), and Natural Language Processing (NLP) to automatically classify emails as spam or non-spam. The goal is to improve email security and user productivity by reducing the presence of unwanted or harmful emails.*#
#CODE

import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load dataset (you can replace this with your own dataset)
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                     sep="\t", names=["label", "message"])
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    return df

# Text cleaning
def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

# Prepare dataset
def prepare_dataset(df):
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    return df

# Train/test split
def split_data(df):
    X = df['cleaned_message']
    y = df['label_num']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

# Evaluate model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on new emails
def predict_email(model, vectorizer, email):
    cleaned = preprocess_text(email)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Main function
def main():
    df = load_data()
    df = prepare_dataset(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model, vectorizer = train_model(X_train, y_train)
    evaluate_model(model, vectorizer, X_test, y_test)

    # Try on new emails
    new_emails = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Please find attached the meeting agenda for tomorrow.",
    ]
    for email in new_emails:
        result = predict_email(model, vectorizer, email)
        print(f"\nEmail: {email}\nPrediction: {result}")

if __name__ == "__main__":
    main()
