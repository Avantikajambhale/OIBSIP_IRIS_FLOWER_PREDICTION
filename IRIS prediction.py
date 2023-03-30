#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df = pd.read_csv("D://Datasets_python//IRIS.csv")


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


df.species.unique()


# In[9]:


import plotly.express as ps
fig = ps.scatter(df,x = "sepal_width" , y = "sepal_length", color = "species")
fig.show()


# In[10]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (12,10))
plt.title('IRIS')
sns.heatmap(df.corr())


# In[ ]:


pip install sklearn


# In[11]:


x = df.drop("species", axis = 1)
y = df["species"]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
k = KNeighborsClassifier(n_neighbors = 1)
k.fit(xtrain, ytrain)


# In[12]:


x_arr = np.array([[3,2,4,0.3]])
per = k.predict(x_arr)
print("Predicted data : {}".format(per))


# In[13]:


x_arr = np.array([[5,4,2,0.9]])
per = k.predict(x_arr)
print("Predicted data : {}".format(per))


# In[14]:


x_arr = np.array([[3,2,4,0.3]])
per = k.predict(x_arr)
print("Predicted data : {}".format(per))


# In[ ]:




