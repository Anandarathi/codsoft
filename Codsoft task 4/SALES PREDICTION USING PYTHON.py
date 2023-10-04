#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv("E:/CODSOFT/advertising.csv")
data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.isnull().sum()


# In[9]:


list(data.columns.values)


# In[10]:


sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales')


# In[11]:


sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',height=7,aspect=0.7,kind='reg')


# In[12]:


feature_cols = ['TV','Radio','Newspaper']
X = np.array(data[feature_cols])
y = np.array(data['Sales'])


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


import statsmodels.api as sm


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[17]:


linreg = LinearRegression()


# In[18]:


linreg.fit(X_train,y_train)


# In[19]:


y_pred = linreg.predict(X_test)


# In[20]:


print(linreg.score(X_test,y_test))


# In[21]:


from sklearn import metrics


# In[22]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[23]:


test_1 = np.array([[230.1,37.8,69.2]])
print(linreg.predict(test_1))


# In[24]:


test_2 = np.array([[151.5,	41.3,	58.5	]])
print(linreg.predict(test_2))


# In[ ]:




