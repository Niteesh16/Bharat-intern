#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_boston


# In[3]:


df=load_boston()


# In[5]:


df["feature_names"]


# In[6]:


x=df["data"]
y=df["target"] 


# In[32]:


from sklearn.linear_model import Ridge


# In[33]:


lr=Ridge(alpha=0.0001)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30)


# In[34]:


lr.fit(x_train,y_train)


# In[35]:


lr.predict(x_test)


# In[36]:


lr.score(x_test,y_test)


# In[ ]:




