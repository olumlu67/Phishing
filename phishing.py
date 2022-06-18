# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 23:04:19 2021

@author: GorkemGoktepe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



df = pd.read_csv("Phishing_Legitimate_full.csv")
df = df.drop(["id","HttpsInHostname"],axis=1)
cr = df.corr()
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=15)
rfc = RandomForestClassifier(n_estimators=100, criterion="gini")
rfc.fit(x_train,y_train)
# print(cr)
y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)   
print(cm)
print("*"*50)
print(classification_report(y_test, y_pred))
print("*"*50)