
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


mooc_df = pd.read_csv('data/HMXPC13_DI_v2_5-14-14.csv')


# In[4]:


mooc_df.columns


# In[5]:


mooc_df.shape


# In[6]:


mooc_df.isnull().sum()


# This workbook is concerned with assembling data to analyze initial dropout. So, we define a set of global variables

# In[7]:


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


# In[8]:


len(mooc_df.course_id.unique()) == len(COURSE_START_DATES.keys())


# Now, we define several helper functions to munge the data into required shape. These functions are called by a master-function that iterates over the list of courses, beats it into the required shape and writes it into a feather file.

# In[9]:


def bin_engaged(row):
    if row['viewed'] == 1 or row['explored'] == 1 or row['certified'] == 1:
        return 1
    else:
        return 0


# In[10]:


def joined_early_by(join_date_offset):
    if join_date_offset < pd.Timedelta(0):
        return -join_date_offset.days
    else:
        return 0


# In[11]:


def joined_late_by(join_date_offset):
    if join_date_offset > pd.Timedelta(0):
        return join_date_offset.days
    else:
        return 0


# In[12]:


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


# In[13]:


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


# In[28]:


def assemble_course_data(mooc_df):
    for id in mooc_df.course_id.unique():
        course = mooc_df['course_id'] == id
        course_df = mooc_df[course]
        
        course_df = course_df[VARS_FOR_ANALYSIS]
        
        print('Munging {}...'.format(id))
        
        course_df['join_date_offset'] = course_df.start_time_DI.apply(pd.to_datetime) -                                         COURSE_START_DATES[id]
            
        course_df = course_df.assign(engaged = course_df.apply(bin_engaged, axis=1),
                                     joined_early_by = course_df.join_date_offset.apply(joined_early_by),
                                     joined_late_by = course_df.join_date_offset.apply(joined_late_by),
                                     age = course_df.YoB.apply(lambda x: 2013 - x),
                                     country = course_df.final_cc_cname_DI.apply(bin_country),
                                     education = course_df.LoE_DI.apply(bin_education))
        
        course_df.drop(['registered', 'viewed', 'explored', 'certified', 'LoE_DI', 'YoB',                         'final_cc_cname_DI', 'start_time_DI', 'join_date_offset'], 
                       axis=1,
                       inplace=True)
        
        course_df_final = pd.get_dummies(course_df, dummy_na=True)
        
        print('Writing {} to file...'.format(id))
        
        course_name = id.split('/')
        
        course_df_final.reset_index().to_feather('processed/{}.feather'.format(''.join(course_name)))


# In[29]:


assemble_course_data(mooc_df)

