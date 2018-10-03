
# coding: utf-8

# In this workbook, we fit the processed raw data to a Logistic Regression model.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc


# In[3]:


from imblearn.over_sampling import SMOTE


# ### Import the data set

# In[4]:


mooc_df = pd.read_feather('../data/processed-mooc-data.feather')


# ### A helper to fit the model

# In[5]:


def fit_model(data_df, classifier_obj, param_grid, n_splits=10, n_repeats=3):
    
    """
    
    Takes in the data and a model object, fits the model using the model object 
    passed in and returns the results from the cross validation process.
    
    It also returns the test set split into features and labels
    
    """
    
    data_clean_df = pd.get_dummies(data_df)
    
    features = np.array(data_clean_df.drop(['engaged'], axis=1))
    labels = np.array(data_clean_df['engaged'])

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=0.2,
                                                                                random_state=20130810)
    
    sm = SMOTE(random_state=20130810)
    
    features_train_smote, labels_train_smote = sm.fit_sample(features_train, labels_train)
    
    grid_search = GridSearchCV(classifier_obj, 
                               param_grid, 
                               cv = RepeatedKFold(n_splits,
                                                  n_repeats,
                                                  random_state=20130810),
                               verbose=1)
    
    grid_search.fit(features_train_smote, labels_train_smote)
    
    return grid_search, features_test, labels_test


# ### Fit the specified logit

# In[6]:


model_logit = LogisticRegression(solver='lbfgs',
                                 n_jobs=-1, 
                                 random_state=20130810,
                                 warm_start=True)


# In[7]:


parameter_grid = { 'C' : [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4] } 


# In[8]:


logit_fit_grid, features_test, labels_test = fit_model(data_df=mooc_df, 
                                                       classifier_obj=model_logit, 
                                                       param_grid=parameter_grid, n_splits=3, n_repeats=1)


# In[9]:


mean_logit_acc = logit_fit_grid.cv_results_['mean_test_score']
sd_logit_acc = logit_fit_grid.cv_results_['std_test_score']


# In[18]:


logit_fit_grid.best_score_


# ### Make predictions on test set and compute ROC and AUC

# In[10]:


preds = logit_fit_grid.predict_proba(features_test)[:, 1]


# In[11]:


fpr, tpr, _ = roc_curve(labels_test, preds)


# In[12]:


pd.DataFrame(dict(false_positive_rate=fpr, true_positive_rate=tpr)).to_feather('../data/roc-logit.feather')


# In[13]:


pd.DataFrame(dict(mean_accuracy=mean_logit_acc, sd_accuracy=sd_logit_acc, auc=auc(fpr, tpr))).to_feather('../data/accuracy-logit.feather')

