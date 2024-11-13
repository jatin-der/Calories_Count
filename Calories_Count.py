#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install xgboost


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[3]:


#Load the data

Calories= pd.read_csv(r"C:\Users\ASUS\Downloads\calories.csv")
Calories


# In[4]:


Calories.head()


# In[5]:


excersice = pd.read_csv(r"C:\Users\ASUS\Downloads\exercise.csv")
excersice


# In[6]:


excersice.head()


# In[7]:


calories_data = pd.concat([excersice,Calories['Calories']],axis=1)


# In[8]:


calories_data.head()


# In[9]:


calories_data.shape


# In[10]:


calories_data.info()


# In[11]:


calories_data.isnull().sum()


# In[12]:


calories_data.describe()


# In[13]:


sns.set()


# In[14]:


sns.countplot(calories_data['Gender'])


# In[15]:


sns.distplot(calories_data['Age'])


# In[16]:


sns.displot(calories_data['Height'])


# In[17]:


sns.distplot(calories_data['Weight'])


# In[18]:


#Correlation
Correlation = calories_data.corr()


# In[19]:


plt.figure(figsize=(10,10))
sns.heatmap(Correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')


# In[20]:


#Converting text data into numerical value
calories_data.replace({"Gender": {"male": 0,"female": 1}}, inplace= True)


# In[21]:


calories_data.head()


# In[25]:


#seprating feature and target
X= calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y= calories_data['Calories']


# In[26]:


print(X)


# In[27]:


print(Y)


# In[28]:


#Splitting data into traning data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[29]:


print(X.shape, X_train.shape, X_test.shape)


# In[ ]:





# In[31]:


#Model Traning
##xgboost
model = XGBRegressor()


# In[32]:


model.fit(X_train, Y_train)


# In[ ]:





# In[33]:


#Evaluation
##Prediction
test_data_prediction = model.predict(X_test)


# In[34]:


print(test_data_prediction)


# In[ ]:





# In[35]:


#Mean Absolute Error
mae = metrics.mean_absolute_error(Y_test,test_data_prediction)


# In[36]:


print("Mean Absolute Error = ", mae)


# In[ ]:




