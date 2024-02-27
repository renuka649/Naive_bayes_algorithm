# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:43:39 2024

@author: renuk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


salary_train=pd.read_csv('SalaryData_Train.csv')
# Assuming you have a DataFrame called 'df' with your dataset

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
df_encoded = salary_train.apply(label_encoder.fit_transform)

# Separate features and target variable
X = df_encoded.drop('Salary', axis=1)
y = df_encoded['Salary']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes model
naive_bayes_model = MultinomialNB()

# Train the model
naive_bayes_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)


# Display evaluation metrics
print(f"Accuracy: {accuracy}")
