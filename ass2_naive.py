# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:38:28 2024

@author: renuk
"""

'''
Prepare a classification model using the Naive Bayes
algorithm the car dataset. 

Use both for model building
1.Business Problem
The model's accuracy and reliability are crucial for making
 informed decisions regarding inventory management, pricing 
 strategies, and sales forecasting. By accurately predicting which
 car models are more likely to sell quickly, the dealership can 
 prioritize marketing efforts, adjust pricing dynamically, and 
 optimize its inventory mix to meet customer demand efficiently.

Ultimately, the goal is to enhance the dealership's competitiveness, increase sales performance, and improve customer satisfaction by ensuring that the available inventory aligns closely with customer preferences and market trends.
1.1 What is business objective?
Optimize Inventory Management: The primary objective is to optimize the dealership's inventory management by predicting which car models are likely to be sold within a specific time frame, such as 30 days. This prediction helps the dealership allocate resources more efficiently by focusing on vehicles that have a higher probability of being sold quickly.

Minimize Holding Costs: By accurately predicting which car models are likely to sell within a short time, the dealership can reduce inventory holding costs associated with keeping vehicles in stock for extended periods. Minimizing holding costs contributes to improving overall profitability and operational efficiency.

Maximize Sales Revenue: The classification model aims to maximize sales revenue by enabling the dealership to prioritize marketing efforts and allocate resources to vehicles with a higher likelihood of being sold quickly. By focusing on high-demand car models, the dealership can capitalize on sales opportunities and generate more revenue.

Improve Pricing Strategies: The predictive model helps the dealership make informed decisions regarding pricing strategies for different car models. By understanding which factors influence the likelihood of a car being sold quickly, the dealership can adjust prices dynamically to attract potential buyers while maintaining profitability.

Data-Driven Decision Making: Foster a culture of data-driven decision-making within the HR department, enabling better strategic planning and resource allocation.

1.1.1 The motivation of human objectives in a business
is to find ways to meet the needs of your employees,
so that they feel valued and supported.
1.1.2 Organic business objectives are goals that 
Maximize-  The primary goal is to maximize sales revenue by
 accurately predicting which car models are likely to be sold
 within a short time frame. By prioritizing marketing efforts 
 and allocating resources effectively, the dealership can increase
 the number of vehicles sold and generate higher revenue.
 
Minimize-Minimize inventory holding costs associated with keeping 
vehicles in stock for extended periods. By accurately predicting 
which car models are likely to sell quickly, the dealership can 
reduce the amount of capital tied up in inventory and minimize 
storage and maintenance expenses.
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

##load the dataset
car_data = pd.read_csv("NB_Car_Ad.csv", encoding='ISO-8859-1')

#cleaning of data

import re
def cleaning_text(i):
    w = []
    i = re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return (" ".join(w))


#Testing above function with some test text

cleaning_text("Hope your are having good week.just checking")
cleaning_text("hope i can understand your feeling 123121.123.hi how are you")
cleaning_text("hi how are you,I am sad")
car_data.text = car_data.text.apply(cleaning_text)
car_data=car_data.loc[car_data.text !="",:]

from sklearn.model_selection import train_test_split
car_train,car_test = train_test_split(car_data, test_size=0.2)

#create  a matrix of token counts for entire text document
def split_into_words(i):
    return[word for word in i.split(" ")]

car_bow = CountVectorizer(analyzer=split_into_words).fit(car_data.text)

all_car_matrix = car_bow.transform(car_data.text)

#for training message
train_car_matrix = car_bow.transform(car_train.text)

#for testing msg
test_car_matrix = car_bow.transform(car_test.text)


#learning term weighting and norming on entire emails

tfidf_transformer = TfidfTransformer().fit(all_car_matrix)


#preparing TFIDF for train mail
train_tfidf = tfidf_transformer.transform(train_car_matrix)

#preparing TFIDF for test mails
test_tfidf = tfidf_transformer.transform(test_car_matrix)
test_tfidf.shape

#now  let us apply this to naive bayes

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_tfidf,car_train.type)

#evaluation on test data

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == car_test.type)
accuracy_test_m