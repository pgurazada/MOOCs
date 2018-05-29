
# coding: utf-8

# In this script we split the data into the training and test set, impute missing values and save the processed data frame back to the folder - `processed-final`

# In[1]:


import os


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import TransformerMixin


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """*Impute missing values*.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[6]:


def split_and_impute(course_df):
    labels = course_df['engaged']
    features = course_df.drop(['index', 'engaged'], axis=1)
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=0.2,
                                                                                random_state=20130810) 
    
    imputer = DataFrameImputer()
    
    imputer.fit(features_train)
    
    return imputer.transform(features_train), imputer.transform(features_test), labels_train, labels_test
    


# Let us now use the scaffolding above to read in and write back the split and imputed data

# In[7]:


# Global variables
DATA_DIR = 'processed/'


# In[8]:


for data_file in os.listdir(DATA_DIR):
    course_df = pd.read_feather(os.path.join(DATA_DIR, data_file))
                                
    print('Processing {}...'.format(data_file))
    
    features_train, features_test, labels_train, labels_test = split_and_impute(course_df)
    
    course_name = data_file[:-8]
    output_dir = os.path.join(DATA_DIR, course_name)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    features_train.reset_index().to_feather(os.path.join(output_dir, 'features_train.feather'))
    features_test.reset_index().to_feather(os.path.join(output_dir, 'features_test.feather'))
    labels_train.reset_index().to_feather(os.path.join(output_dir, 'labels_train.feather'))
    labels_test.reset_index().to_feather(os.path.join(output_dir, 'labels_test.feather'))

