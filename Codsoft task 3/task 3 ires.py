#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libary


# In[18]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[51]:


df = pd.read_csv("iris.csv")


# In[52]:


df


# In[53]:


df['Species'].value_counts()
#Counts the number of inputs cooresponding to each Species


# In[21]:


# Data Preprocesing


# In[23]:


df.isnull()


# In[24]:


df.isnull().sum()


# In[25]:


#Display basic statistics about the data
df.describe().transpose()


# In[26]:


df.isnull().any()


# In[31]:


# Display the number of samples for each class
df['species_setosa'].value_counts()


# In[33]:


df['species_versicolor'].value_counts()


# In[54]:


#Exploratory data analysis


# In[38]:


# Plot histograms of each feature
df['sepal_length'].hist(color='red')


# In[42]:


df['sepal_width'].hist(color='blue')


# In[43]:


df['petal_length'].hist(color='green')


# In[44]:


#Plotting the histogram of all features toghether
df['sepal_length'].hist()
df['sepal_width'].hist()
df['petal_length'].hist()
df['petal_width'].hist()


# In[45]:


# Plot scatterplots to visualize relationships between features
colors = ['red', 'green', 'blue']
species = [0, 1, 2]


# In[56]:


plt.figure(figsize=(5, 3))
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette=['blue', 'green', 'red'])
plt.xlabel('Sepal Length in Cm')
plt.ylabel('Sepal Width in Cm')
plt.title('Sepal Length vs Sepal Width')
plt.show()


# In[57]:


plt.figure(figsize=(5,3))
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette=['blue', 'green', 'red'])
plt.xlabel('Petal Length in Cm')
plt.ylabel('Petal Width in Cm')
plt.title('Petal Length vs Petal Width')
plt.show()


# In[62]:


sns.pairplot(df,hue='Species')


# In[63]:


#Correlation Matrix


# In[64]:


# Compute the correlation matrix 
df.corr().transpose()


# In[65]:


# display the correlation matrix using a heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, ax=ax, cmap='YlGnBu')


# In[58]:


#Splitting data set to train and test data
from sklearn.model_selection import train_test_split

x = df.drop(columns=["Species"])
y = df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[61]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
logreg_Accuracy = logmodel.score(x_test, y_test) * 100
print("Accuracy (Logistic Regression): ", logreg_Accuracy)


# In[ ]:




