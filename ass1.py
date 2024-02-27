# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:38:28 2024

@author: renuk
"""

##assignments
##SalaryData_Train, SalaryData_Test

'''
prepare classification model using naive bayes algorithm for the salary dataset 
Train and test datasets are given seperately.Use both for model building. 

Business Objectives:
Predict Salary Levels:Develop a classification model to predict whether an 
individual's salary is above or below a specified threshold.
Identify Key Features:Identify the most important features that contribute to
the prediction of salary levels.
Business Contriant:
 

age
workclass	
education	
educationno	maritalstatus	
occupation	
relationship	
race	
sex	
capitalgain	
capitalloss	
hoursperweek	
native	
Salary

'''


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import accuracy_score

salary_train=pd.read_csv('SalaryData_Train.csv')
salary_test=pd.read_csv('SalaryData_Test.csv')

# Assuming the dataset is loaded into a DataFrame named 'df'
# Convert categorical features to numerical using encoding (e.g., one-hot encoding)
df_encoded = pd.get_dummies(salary_train, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])
df_encoded1 = pd.get_dummies(salary_test, columns=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])

# Separate features and target variable
X = df_encoded.drop('Salary', axis=1)
y= df_encoded['Salary']

X1= df_encoded1.drop('Salary', axis=1)
y1= df_encoded1['Salary']

# Split the data into training and testing sets
X_train, X1_test, y_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Assuming 'text_feature' is a text column, and we use CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['text_feature'])
X1_test_vectorized = vectorizer.transform(X1_test['text_feature'])

# Initialize Naive Bayes model
naive_bayes_model = MB()

# Train the model
naive_bayes_model.fit(X_train_vectorized, y_train)

# Make predictions on the test data
predictions = naive_bayes_model.predict(X1_test_vectorized)
# Evaluate the model
accuracy = accuracy_score(y1_test, predictions)
print(f"Accuracy: {accuracy}")






