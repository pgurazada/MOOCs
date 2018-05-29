
# coding: utf-8

# Based on preliminary analysis of the data, we conclude that using random forest classifiers is the best way to classify drop-out. Given the imbalance in classes and the importance of misclassification of drop-outs, we use the Kappa estimator as  the benchmark to train the model.

# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer

from imblearn.over_sampling import SMOTE


# In[ ]:


def compute_kappa(course_df):
    data_clean = pd.get_dummies(course_df)
        
    labels = np.array(data_clean.engaged)
    features = np.array(data_clean.drop('engaged', axis = 1))
        
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                                train_size = 0.8, 
                                                                                random_state = 20130810)
    
    sm = SMOTE(random_state = 20130810)
    features_train_smote, labels_train_smote = sm.fit_sample(features_train, labels_train) 
        
    grid_search = GridSearchCV(RandomForestClassifier(n_jobs = 3, 
                                                      n_estimators = 500,
                                                      warm_start = True,
                                                      random_state = 20130810),
                               param_grid = {'max_features': [6, 8, 10, 12]},
                               cv = RepeatedKFold(n_splits = 10, 
                                                  n_repeats = 3, 
                                                  random_state=20130810),
                               scoring = make_scorer(cohen_kappa_score))
        
    grid_search.fit(features_train_smote, labels_train_smote)
        
    kappa_train = max(grid_search.cv_results_['mean_test_score'])
        
    kappa_test = cohen_kappa_score(grid_search.best_estimator_.predict(features_test), labels_test)
    
    cv_result = grid_search.cv_results_
    
    all_splits_result = [max(cv_result[k]) for k in cv_result.keys() if 'split' in k and 'test_score' in k]
    
    return kappa_train, min(all_splits_result), max(all_splits_result), kappa_test


# In[ ]:


course_metrics = {"course_name": [], 
                  "kappa_train": [], 
                  "min_kappa_train": [],
                  "max_kappa_train": [],
                  "kappa_test": []}


# ### 1. CB22x - The Ancient Greek Hero

# In[2]:


cb22x = pd.read_feather("data/HarvardX_CB22x_2013_Spring.feather")


# In[7]:


cb22x.education.value_counts()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(cb22x)')


# In[ ]:


course_metrics["course_name"].append('HarvardCB22x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 2. CS50x - Introduction to Computer Science I

# In[ ]:


cs50x = pd.read_feather("data/HarvardX_CS50x_2012.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(cs50x)')


# In[ ]:


course_metrics["course_name"].append('HarvardCS50x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 3. ER22x - Justice

# In[ ]:


er22x = pd.read_feather("data/HarvardX_ER22x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(er22x)')


# In[ ]:


course_metrics["course_name"].append('HarvardER22x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 4. PH207x - Health in Numbers: Quantitative Methods in Clinical & Public Health Research

# In[ ]:


ph207x = pd.read_feather("data/HarvardX_PH207x_2012_Fall.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(ph207x)')


# In[ ]:


course_metrics["course_name"].append('HarvardPH207x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 5. PH278x - Human Health and Global Environmental Change

# In[ ]:


ph278x = pd.read_feather("data/HarvardX_PH278x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(ph278x)')


# In[ ]:


course_metrics["course_name"].append('HarvardPH278x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 6. 6.002x (Fall) - Circuits and Electronics

# In[ ]:


mit6002x = pd.read_feather("data/MITx_6_002x_2012_Fall.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit6002x)')


# In[ ]:


course_metrics["course_name"].append('MIT6002x_Fall')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 7. 6.002x (Spring) - Circuits and Electronics

# In[ ]:


mit6002x = pd.read_feather("data/MITx_6_002x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit6002x)')


# In[ ]:


course_metrics["course_name"].append('MIT6002x_Spring')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 8. 14.73x - The Challenges of Global Poverty

# In[ ]:


mit1473x = pd.read_feather("data/MITx_14_73x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit1473x)')


# In[ ]:


course_metrics["course_name"].append('MIT1473x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 9. 2.01x - Elements of Structures

# In[ ]:


mit201x = pd.read_feather("data/MITx_2_01x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit201x)')


# In[ ]:


course_metrics["course_name"].append('MIT201x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 10. 3.091x(Fall) - Introduction to Solid State Chemistry

# In[ ]:


mit3091x = pd.read_feather("data/MITx_3_091x_2012_Fall.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit3091x)')


# In[ ]:


course_metrics["course_name"].append('MIT3091x_Fall')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 11. 3.091x (Spring) - Introduction to Solid State Chemistry

# In[ ]:


mit3091x = pd.read_feather("data/MITx_3_091x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit3091x)')


# In[ ]:


course_metrics["course_name"].append('MIT3091x_Spring')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 12. 6.00x (Fall) - Introduction to Computer Science and Programming

# In[ ]:


mit600x = pd.read_feather("data/MITx_6_00x_2012_Fall.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit600x)')


# In[ ]:


course_metrics["course_name"].append('MIT600x_Fall')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 13. 6.00x (Spring) - Introduction to Computer Science and Programming

# In[ ]:


mit600x = pd.read_feather("data/MITx_6_00x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit600x)')


# In[ ]:


course_metrics["course_name"].append('MIT600x_Spring')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 14. 8.02x - Electricity and Magnetism

# In[ ]:


mit802x = pd.read_feather("data/MITx_8_02x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit802x)')


# In[ ]:


course_metrics["course_name"].append('MIT802x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 15. 7.00x - Introduction to Biology - The Secret of Life

# In[ ]:


mit700x = pd.read_feather("data/MITx_7_00x_2013_Spring.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit700x)')


# In[ ]:


course_metrics["course_name"].append('MIT700x')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# ### 16. 8.MReVx - Mechanics ReView

# In[ ]:


mit8mrevx = pd.read_feather("data/MITx_8_MReV_2013_Summer.feather")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'best_k_train, min_k_train, max_k_train, k_test = compute_kappa(mit8mrevx)')


# In[ ]:


course_metrics["course_name"].append('MIT8MReVx')
course_metrics["kappa_train"].append(best_k_train)
course_metrics["min_kappa_train"].append(min_k_train)
course_metrics["max_kappa_train"].append(max_k_train)
course_metrics["kappa_test"].append(k_test)


# In[ ]:


course_kappa = pd.DataFrame(course_metrics)


# In[ ]:


course_kappa.to_feather("data/course_kappa.feather")

