# -*- coding: utf-8 -*-
"""
Created on Mon feb 1 23:15:57 2024

@author: renuk
"""


'''
Prepare a classification model using the Naive Bayes
algorithm the Disaster dataset. 

Use both for model building
1.Business Problem
The model's accuracy and reliability are crucial for ensuring timely
 and appropriate responses to emergency situations. By accurately 
 categorizing incoming messages, the organization can prioritize its 
 response efforts, dispatch resources to the areas most in need, and coordinate rescue and relief operations more effectively.

Ultimately, the goal is to improve the organization's capacity to manage and respond to disasters by leveraging machine learning techniques to analyze and classify incoming messages, thereby enhancing overall emergency response capabilities and potentially saving lives and mitigating damage during critical situations.

1.1 What is business objective?
Timely Response and Resource Allocation: The primary objective is to enable the organization to respond promptly to incoming messages by categorizing them accurately. By quickly identifying urgent requests for help, reports of incidents, or critical information, the organization can allocate resources and prioritize its response efforts more effectively.

Optimize Resource Utilization: By categorizing messages based on their urgency and nature, the organization can optimize the allocation of resources such as personnel, equipment, and supplies. This ensures that resources are directed to the areas and situations that require immediate attention, thereby maximizing the impact of the organization's response efforts.

Improve Situational Awareness: Another objective is to enhance the organization's situational awareness by systematically categorizing and analyzing incoming messages. By understanding the types and distribution of messages received during a disaster event, the organization can gain insights into the evolving situation, identify emerging needs and trends, and make informed decisions to adapt its response strategy accordingly.
1.1.1 The motivation of human objectives in a business
is to find ways to meet the needs of your employees,
so that they feel valued and supported.

1.1.2 Organic business objectives are goals that 

Maximize-  The primary goal is to maximize the accuracy of the 
classification model in categorizing incoming messages correctly.
 A higher accuracy rate ensures that the organization can effectively
 identify and prioritize urgent requests for help, incident reports,
 and critical information during disaster events.
 
Minimize-Minimize misclassification errors, including false positives
 and false negatives, in the classification model predictions. 
 False positives may result in resources being allocated to
 non-urgent situations, while false negatives may lead to delays in
 responding to critical incidents. Minimizing misclassification 
 errors enhances the reliability and effectiveness of the response
 efforts.
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

##load the dataset
Disaster_tweets_data= pd.read_csv("Disaster_tweets_NB.csv", encoding='ISO-8859-1')

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
Disaster_tweets_data.text = Disaster_tweets_data.text.apply(cleaning_text)
Disaster_tweets_data=Disaster_tweets_data.loc[Disaster_tweets_data.text !="",:]

from sklearn.model_selection import train_test_split
Disaster_tweets_train,Disaster_tweets_test = train_test_split(Disaster_tweets_data, test_size=0.2)

#create  a matrix of token counts for entire text document
def split_into_words(i):
    return[word for word in i.split(" ")]

Disaster_tweets_bow = CountVectorizer(analyzer=split_into_words).fit(Disaster_tweets_data.text)

all_Disaster_tweets_matrix = Disaster_tweets_bow.transform(Disaster_tweets_data.text)

#for training message
train_Disaster_tweets_matrix = Disaster_tweets_bow.transform(Disaster_tweets_train.text)

#for testing msg
test_Disaster_tweets_matrix = Disaster_tweets_bow.transform(Disaster_tweets_test.text)


#learning term weighting and norming on entire emails

tfidf_transformer = TfidfTransformer().fit(all_Disaster_tweets_matrix)


#preparing TFIDF for train mail
train_tfidf = tfidf_transformer.transform(train_Disaster_tweets_matrix)

#preparing TFIDF for test mails
test_tfidf = tfidf_transformer.transform(test_Disaster_tweets_matrix)
test_tfidf.shape

#now  let us apply this to naive bayes

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_tfidf,Disaster_tweets_train.type)

#evaluation on test data

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == Disaster_tweets_test.type)
accuracy_test_m