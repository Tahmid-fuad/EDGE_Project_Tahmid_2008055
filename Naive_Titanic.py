#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Tutorial Part 1: Predicting survival from titanic crash

# In[9]:


import pandas as pd

df = pd.read_csv(r"tested.csv")
df.head()


# In[10]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head(7)


# In[11]:


df.shape


# In[12]:


inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[13]:


inputs


# In[14]:


inputs.isnull().sum()


# In[15]:


inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
inputs


# In[16]:


inputs.Age[:10]


# In[17]:


inputs.isnull().sum()


# In[18]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[19]:


inputs.Fare = inputs.Fare.fillna(inputs.Fare.mean())
inputs.head()


# In[20]:


inputs.shape


# In[21]:


inputs.isnull().sum()


# In[22]:


target


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


X_test


# In[27]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB() ## when data-distribution is Normal


# In[28]:


model.fit(X_train,y_train)


# In[29]:


model.score(X_test,y_test)


# In[30]:


X_test[0:10]


# In[31]:


# comparing wuth y_test
y_test[0:10]


# In[32]:


model.predict(X_test[0:10])


# In[ ]:




