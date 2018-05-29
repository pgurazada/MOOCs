
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style('ticks')
plt.grid(False)


# In[4]:


mooc_df = pd.read_csv('data/HMXPC13_DI_v2_5-14-14.csv')


# In[5]:


mooc_df.columns


# In[6]:


mooc_df.shape


# In[7]:


mooc_df.isnull().sum()


# In[ ]:


course = mooc_df['course_id'] == 'HarvardX/CB22x/2013_Spring'
cb22x_df = mooc_df[course]


# This workbook is concerned with assembling data to analyze initial dropout. So, we define a set of global variables

# In[14]:


VARS_FOR_ANALYSIS = ['registered', 'viewed', 'explored', 'certified', 'gender', 'LoE_DI',                      'YoB', 'final_cc_cname_DI', 'start_time_DI']

COURSE_START_DATES = {'HarvardX/CS50x/2012' : pd.to_datetime('2012-10-15'),
                      'HarvardX/CB22x/2013_Spring' : pd.to_datetime('2013-03-13'),
                      'HarvardX/ER22x/2013_Spring' : pd.to_datetime('2013-03-02'),
                      'HarvardX/PH207x/2012_Fall' : pd.to_datetime('2012-10-15'),
                      'HarvardX/PH278x/2013_Spring' : pd.to_datetime('2013-05-15'),
                      'MITx/14.73x/2013_Spring' : pd.to_datetime('2013-02-12'),
                      'MITx/2.01x/2013_Spring' : pd.to_datetime('2013-04-15'),
                      'MITx/6.002x/2012_Fall' : pd.to_datetime('2012-09-05'),
                      'MITx/6.002x/2013_Spring' : pd.to_datetime('2013-03-03'),
                      'MITx/3.091x/2012_Fall' : pd.to_datetime('2012-10-09'),
                      'MITx/3.091x/2013_Spring' : pd.to_datetime('2013-02-05'),
                      'MITx/6.00x/2012_Fall' : pd.to_datetime('2012-09-06'),
                      'MITx/6.00x/2013_Spring' : pd.to_datetime('2013-02-04'),
                      'MITx/7.00x/2013_Spring' : pd.to_datetime('2013-03-05'),
                      'MITx/8.02x/2013_Spring' : pd.to_datetime('2013-02-18'),
                      'MITx/8.MReV/2013_Summer' : pd.to_datetime('2013-06-01')}


# In[15]:


len(mooc_df.course_id.unique()) == len(COURSE_START_DATES.keys())


# In[ ]:


cb22x_df = cb22x_df[VARS_FOR_ANALYSIS]


# In[ ]:


cb22x_df.columns


# Now, we define several helper functions to munge the data into required shape

# In[ ]:


def bin_engaged(row):
    if row['viewed'] == 1 or row['explored'] == 1 or row['certified'] == 1:
        return 1
    else:
        return 0


# In[ ]:


def joined_early_by(join_date_offset):
    if join_date_offset < pd.Timedelta(0):
        return -join_date_offset.days
    else:
        return 0


# In[ ]:


def joined_late_by(join_date_offset):
    if join_date_offset > pd.Timedelta(0):
        return join_date_offset.days
    else:
        return 0


# In[ ]:


def bin_country(country_name):
    if country_name == "United States":
        return 'US'
    elif country_name in ["India", "Pakistan", "Bangladesh", "China", "Indonesia", "Japan", "Other East Asia",                           "Other Middle East/Central Asia", "Other South Asia", "Philippines", "Egypt"]: 
        return 'AS'
    elif country_name in ["France", "Germany", "Greece", "Other Europe", "Poland", "Portugal", "Russian Federation",                           "Spain", "Ukraine", "United Kingdom"]:
        return 'EU'
    elif country_name in ["Morocco", "Nigeria", "Other Africa"]:
        return 'AF'
    else:
        return 'OT'


# In[ ]:


def bin_education(LoE_DI):
    if LoE_DI == "Less than Secondary":
        return 'LS'
    elif LoE_DI == "Secondary":
        return 'SE'
    elif LoE_DI == "Bachelor's":
        return 'BA'
    elif LoE_DI == "Master's":
        return 'MA'
    else:
        return 'DO'


# In[ ]:


cb22x_df['join_date_offset'] = cb22x_df.start_time_DI.apply(pd.to_datetime) -                                COURSE_START_DATES['HarvardX/CB22x/2013_Spring']


# In[ ]:


cb22x_df = cb22x_df.assign(engaged = cb22x_df.apply(bin_engaged, axis=1),
                           joined_early_by = cb22x_df.join_date_offset.apply(joined_early_by),
                           joined_late_by = cb22x_df.join_date_offset.apply(joined_late_by),
                           age = cb22x_df.YoB.apply(lambda x: 2013 - x),
                           country = cb22x_df.final_cc_cname_DI.apply(bin_country),
                           education = cb22x_df.LoE_DI.apply(bin_education))


# In[ ]:


cb22x_df = cb22x_df.drop(['registered', 'viewed', 'explored', 'certified', 'LoE_DI', 'YoB',                           'final_cc_cname_DI', 'start_time_DI', 'join_date_offset'], axis=1)


# In[ ]:


cb22x_df.head()


# In[ ]:


cb22x_df.info()


# In[ ]:


cb22x_df_final = pd.get_dummies(cb22x_df, dummy_na=True)


# In[ ]:


cb22x_df_final.info()

