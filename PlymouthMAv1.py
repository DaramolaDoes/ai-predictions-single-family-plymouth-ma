#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd


# In[18]:


# Importing CSV and defining columns
df = pd.read_csv('C:/Users/olubo/Documents/plymouthMAO.csv')


# In[19]:


#view data
df


# In[20]:


import statsmodels.api as sm


# In[22]:


#define response variable
y = df['price']


# In[23]:


#define predictor variables
x = df['date']


# In[24]:


#add constant to predictor variables
x = sm.add_constant(x)


# In[30]:


#fit linear regression model
model = sm.OLS(y, x).fit()


# In[31]:


#view model summary
print(model.summary())


# In[33]:


#import necessary libraries 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[34]:


#fit simple linear regression model
model = ols('date ~ price', data=df).fit()


# In[35]:


#view model summary
print(model.summary())


# In[36]:


#define figure size
fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'price', fig=fig)


# In[ ]:




