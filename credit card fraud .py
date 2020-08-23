#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv(r'C:\Users\Vikas Patel\Downloads\creditcard.csv')

data.head()

fraud = data.loc[data['Class'] == 1]
regular = data.loc[data['Class'] == 0]

from sklearn import linear_model
from sklearn.model_selection import train_test_split

x = data.iloc[:,:-1]
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .35)

clf = linear_model.LogisticRegression(C=1e5)

clf.fit(x_train, y_train)

y_predict = np.array(clf.predict(x_test))
y = np.array(y_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test,y_predict ))

print(accuracy_score(y_test, y_predict))

print(classification_report(y_test,y_predict))

