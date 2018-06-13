
# coding: utf-8

# Based on preliminary analysis of the data, we conclude that using random forest classifiers is the best way to classify drop-out. Given the imbalance in classes and the importance of misclassification of drop-outs, we use the F1 score as  the benchmark to train the model.

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


# Algorithm

from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Model selection metrics

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score


# In[4]:


# Tools to save the best models to disk

from sklearn.externals import joblib


# In[5]:


DATA_DIR = 'processed/'
COURSE_DIR_LIST = [d[0] for d in os.walk(DATA_DIR)][1:]


# In[6]:


course_metrics = {'course_name': [], 
                  'F1_train_mean': [],
                  'F1_train_min' : [],
                  'F1_train_max' : [],
                  'F1_test': []}


# In[7]:


# A general parameter space that will be run across all the data sets for tuning

pgrid = {'min_samples_leaf' : np.arange(20, 200, 25),
         'min_samples_split' : np.arange(20, 200, 25),
         'max_depth' : np.arange(25, 200, 50),
         'n_estimators' : np.arange(100, 2000, 200)}


# We then define the random forest classifier object and the grid search object that will be fit to the data

# In[8]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[9]:


grid_cv = RandomizedSearchCV(estimator=rf_classif,
                             param_distributions=pgrid,
                             scoring={'F1' : 'f1',
                                      'Precision' : 'precision',
                                      'Recall' : 'recall'},
                             refit='F1',
                             n_iter=10,
                             cv=5,
                             random_state=20130810,
                             n_jobs=-1)


# In[10]:


def features_from_data(course_dir):
    '''
    This function takes a string of the directory where the features and labels data of a course lies, and converts
    them into the feature matrix and the output vector.
    
    It returns the train and test data
    '''
    features_train = pd.read_feather(course_dir + '/features_train.feather').drop('index', axis=1).as_matrix()
    features_test = pd.read_feather(course_dir + '/features_test.feather').drop('index', axis=1).as_matrix()
    
    labels_train = pd.read_feather(course_dir + '/labels_train.feather').drop('index', axis=1).values.ravel()
    labels_test = pd.read_feather(course_dir + '/labels_test.feather').drop('index', axis=1).values.ravel()
    
    return features_train, features_test, labels_train, labels_test    


# In[11]:


for course_dir in COURSE_DIR_LIST:
    features_train, features_test, labels_train, labels_test = features_from_data(course_dir)
    
    print('Optimizing hyperparameters for {}'.format(course_dir))
    
    get_ipython().magic('time grid_cv.fit(features_train, labels_train)')
    
    F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)
    
    course_metrics['course_name'].append(course_dir[10:]) # drop parent folder name
    course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
    course_metrics['F1_train_min'].append(grid_cv.cv_results_['mean_test_F1'].min())
    course_metrics['F1_train_max'].append(grid_cv.cv_results_['mean_test_F1'].max())
    course_metrics['F1_test'].append(F1_test_score)
    
    print('Pickling {}'.format(course_dir))
    
    joblib.dump(grid_cv.best_estimator_, course_dir[10:] + '.pkl')


# In[16]:


pd.DataFrame(course_metrics).reset_index().to_csv('course_metrics.csv')

