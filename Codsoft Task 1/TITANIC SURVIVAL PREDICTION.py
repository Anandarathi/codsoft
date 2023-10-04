#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


ts=pd.read_csv("E:/CODSOFT/tested.csv")
ts


# In[79]:


ts.shape


# In[80]:


ts.info()


# In[81]:


ts.isnull().sum()


# In[82]:


ts= ts.drop(columns='Cabin', axis=1)


# In[83]:


ts['Age'].fillna(ts['Age'].mean(), inplace=True)


# In[84]:


print(ts['Fare'].mode())


# In[85]:


print(ts['Fare'].mode()[0])


# In[86]:


ts['Fare'].fillna(ts['Fare'].mode()[0], inplace=True)


# In[87]:


ts.isnull().sum()


# In[88]:


ts.describe()


# In[89]:


ts['Survived'].value_counts()


# In[90]:


sns.set()


# In[91]:


sns.countplot(x='Survived', data=ts)


# In[92]:


sns.countplot(x='Sex', data=ts)


# In[93]:


sns.countplot(x='Sex', hue='Survived', data=ts)


# In[94]:


sns.countplot(x='Pclass', data=ts)


# In[95]:


sns.countplot(x='Pclass', hue='Survived', data=ts)


# In[96]:


ts['Sex'].value_counts()


# In[97]:


ts['Embarked'].value_counts()


# In[98]:


ts.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[99]:


ts.head()


# In[100]:


X = ts.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = ts['Survived']
print(X)


# In[101]:


print(Y)


# In[102]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[103]:


print(X.shape, X_train.shape, X_test.shape)


# In[105]:


model.fit(X_train, Y_train)


# In[106]:


X_train_prediction = model.predict(X_train)
print(X_train_prediction)


# In[107]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[108]:


X_test_prediction = model.predict(X_test)
print(X_test_prediction)


# In[109]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




