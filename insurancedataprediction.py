# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:56:46 2019

@author: Yadav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
def linearmodel(ty):
    X=ty[['bmi','smoker','sex']]
    Y=ty[['charges']]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.34,random_state=20)
    model=LinearRegression()
    model.fit(X_train,Y_train)
#predicting output
    #model.coef_ this is for coefficient of attributes in linear regression
    #model.intercept_ this is for intercept value
    Y_predict=model.predict(X_test)
    print(Y_predict)
    result=metrics.mean_squared_error(Y_test,Y_predict)
    print("the rmse value is")
    print(np.sqrt(result))
    print("the r2 value is")    
    res=r2_score(Y_test,Y_predict)
    print(res)
    plt.plot(X_test,Y_test)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.show()
    plt.plot(X_test,Y_predict)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.show()

    
def cleandata(x):
    data_1=x[['sex']].replace('female',0)#replacing categorical values with numeric in gender 0=female 1=male
    data_1=data_1[['sex']].replace('male',1)#create column changing nominal to binary
    y=x.drop('sex',axis=1)#drop column sex having male and female
    y.insert(2,'sex',data_1)# add column sex having male 1 female 0
    data_2=x[['smoker']].replace('yes',1)#replace categorical values with numeric in smoker 1=yes 0=no
    data_2=data_2[['smoker']].replace('no',0)
    z=y.drop('smoker',axis=1)# drop column with smoker yes or no
    z.insert(5,'smoker',data_2)#add smoker data where yes=1 no=0
    return z     
location="C:\\Users\\Yadav\\Desktop\\practice\\insurance.csv"
data=pd.read_csv(location)
cv=cleandata(data)
linearmodel(cv)