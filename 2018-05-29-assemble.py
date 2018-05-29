
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style('ticks')
plt.grid(False)


# In[5]:


mooc_df = pd.read_csv('data/HMXPC13_DI_v2_5-14-14.csv')


# In[8]:


mooc_df.columns


# In[10]:


mooc_df.shape


# In[9]:


mooc_df.isnull().sum()


# In[12]:


mooc_df.course_id.unique()


# In[16]:


course = mooc_df['course_id'] == 'HarvardX/CB22x/2013_Spring'
cb22x_df = mooc_df[course]


# This workbook is concerned with assembling data to analyze initial dropout. So, we define a set of global variables

# In[34]:


VARS_FOR_INITIAL_DROPOUT = ['registered', 'viewed', 'explored', 'certified', 'gender', 'LoE_DI',                             'YoB', 'final_cc_cname_DI', 'start_time_DI']

COURSE_START_DATES = {'HarvardX/CS50x/2012' : pd.to_datetime('2012-10-15'),
                      'HarvardX/CB22x/2013_Spring' : pd.to_datetime('2013-03-13')}


# In[19]:


cb22x_df = cb22x_df[VARS_FOR_INITIAL_DROPOUT]


# In[20]:


cb22x_df.columns


# Now, we define several helper functions to munge the data into required shape

# In[26]:


def categorize_engaged(row):
    if row['viewed'] == 1 or row['explored'] == 1 or row['certified'] == 1:
        return 1
    else:
        return 0


# In[28]:


cb22x_df = cb22x_df.assign(engaged = cb22x_df.apply(categorize_engaged, axis=1))

