
# coding: utf-8

# In this workbook, we fit the processed raw data to narrow down the parameter space of the XGBoost Model

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


import xgboost as xgb


# In[11]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV


# In[5]:


from imblearn.over_sampling import SMOTE


# In[6]:


mooc_df = pd.read_feather('../data/processed-mooc-data.feather')

mooc_clean_df = pd.get_dummies(mooc_df)

features = np.array(mooc_clean_df.drop('engaged', axis=1))
labels = np.array(mooc_clean_df['engaged'])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.2,
                                                                            random_state=20130810)

sm = SMOTE(random_state=20130810)
    
features_train_smote, labels_train_smote = sm.fit_sample(features_train, labels_train)


# In[7]:


train_data = xgb.DMatrix(features_train_smote, label=labels_train_smote)
test_data = xgb.DMatrix(features_test, label=labels_test)


# In[8]:


params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:logistic',
    'eval_metric' : ['auc', 'error']
}


# In[9]:


cv_results = xgb.cv(params, 
                    train_data, 
                    num_boost_round=999, 
                    metrics={'error', 'auc'},
                    early_stopping_rounds=10,
                    nfold=5,
                    seed=20130810)


# In[10]:


cv_results


# In[29]:


model_xgb = xgb.XGBClassifier(objective='reg:logistic')


# In[30]:


random_grid = {'n_estimators': [200, 300, 400],
               'max_depth': [6, 10, 12]}


# In[31]:


model_random_grid = RandomizedSearchCV(model_xgb, 
                                       random_grid, 
                                       n_iter=5, 
                                       cv=3, 
                                       verbose=1, 
                                       random_state=20130810,
                                       n_jobs=-1)


# In[32]:


model_random_grid.fit(features_train_smote, labels_train_smote)


# In[33]:


model_random_grid.best_params_


# In[17]:


pd.DataFrame(model_random_grid.best_params_).to_feather('../data/xgb_best_params_random.feather')

