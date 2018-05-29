
# coding: utf-8

# Based on preliminary analysis of the data, we conclude that using random forest classifiers is the best way to classify drop-out. Given the imbalance in classes and the importance of misclassification of drop-outs, we use the Kappa estimator as  the benchmark to train the model.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Algorithm

from sklearn.ensemble import RandomForestClassifier


# In[49]:


# Model selection metrics

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score


# In[55]:


course_metrics = {'course_name': [], 
                  'F1_train_mean': [],
                  'F1_train_sd' : [],
                  'F1_test': []}


# ### 1. CB22x - The Ancient Greek Hero

# In[5]:


cb22x_features_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_train.feather')
cb22x_features_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_test.feather')
cb22x_labels_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_train.feather')
cb22x_labels_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_test.feather')


# In[34]:


features_train, features_test = cb22x_features_train.drop('index', axis=1), cb22x_features_test.drop('index', axis=1)
labels_train, labels_test = cb22x_labels_train.drop('index', axis=1), cb22x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[36]:


pgrid = {'min_samples_leaf' : [20, 50, 100],
         'min_samples_split' : [20, 30, 100],
         'max_depth' : [25, 100, 200],
         'n_estimators' : [500, 1000]}


# In[37]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[38]:


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


# In[39]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[40]:


grid_cv.best_estimator_


# In[41]:


grid_cv.best_score_


# In[52]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[56]:


course_metrics['course_name'].append('HarvardCB22x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 2. CS50x - Introduction to Computer Science I

# In[58]:


cs50x_features_train = pd.read_feather('processed-final/HarvardXCS50x2012_features_train.feather')
cs50x_features_test = pd.read_feather('processed-final/HarvardXCS50x2012_features_test.feather')
cs50x_labels_train = pd.read_feather('processed-final/HarvardXCS50x2012_labels_train.feather')
cs50x_labels_test = pd.read_feather('processed-final/HarvardXCS50x2012_labels_test.feather')


# In[59]:


features_train, features_test = cs50x_features_train.drop('index', axis=1), cs50x_features_test.drop('index', axis=1)
labels_train, labels_test = cs50x_labels_train.drop('index', axis=1), cs50x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[ ]:


pgrid = {'min_samples_leaf' : [20, 50, 100],
         'min_samples_split' : [20, 30, 100],
         'max_depth' : [25, 100, 200],
         'n_estimators' : [500, 1000]}


# In[ ]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[ ]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[ ]:


grid_cv.best_estimator_


# In[41]:


grid_cv.best_score_


# In[52]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


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

# In[62]:


er22x_features_train = pd.read_feather('processed-final/HarvardXER22x2013_Spring_features_train.feather')
er22x_features_test = pd.read_feather('processed-final/HarvardXER22x2013_Spring_features_test.feather')
er22x_labels_train = pd.read_feather('processed-final/HarvardXER22x2013_Spring_labels_train.feather')
er22x_labels_test = pd.read_feather('processed-final/HarvardXER22x2013_Spring_labels_test.feather')


# In[63]:


features_train, features_test = er22x_features_train.drop('index', axis=1), er22x_features_test.drop('index', axis=1)
labels_train, labels_test = er22x_labels_train.drop('index', axis=1), er22x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[65]:


pgrid = {'min_samples_leaf' : [20, 50, 100],
         'min_samples_split' : [20, 30, 100],
         'max_depth' : [25, 100, 200],
         'n_estimators' : [500, 1000]}


# In[66]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[67]:


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


# In[68]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[69]:


grid_cv.best_estimator_


# In[70]:


grid_cv.best_score_


# In[71]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[73]:


course_metrics['course_name'].append('HarvardER22x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 4. PH207x - Health in Numbers: Quantitative Methods in Clinical & Public Health Research

# In[74]:


ph207x_features_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_train.feather')
ph207x_features_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_test.feather')
ph207x_labels_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_train.feather')
ph207x_labels_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_test.feather')


# In[75]:


features_train, features_test = ph207x_features_train.drop('index', axis=1), ph207x_features_test.drop('index', axis=1)
labels_train, labels_test = ph207x_labels_train.drop('index', axis=1), ph207x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[81]:


pgrid = {'min_samples_leaf' : [50, 100, 200],
         'min_samples_split' : [20, 30, 100],
         'max_depth' : [25, 100, 200],
         'n_estimators' : [500, 1000]}


# In[82]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[83]:


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


# In[84]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[85]:


grid_cv.best_estimator_


# In[86]:


grid_cv.best_score_


# In[87]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[89]:


course_metrics['course_name'].append('HarvardPH207x_2012_Fall')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 5. PH278x - Human Health and Global Environmental Change

# In[91]:


ph278x_features_train = pd.read_feather('processed-final/HarvardXPH278x2013_Spring_features_train.feather')
ph278x_features_test = pd.read_feather('processed-final/HarvardXPH278x2013_Spring_features_test.feather')
ph278x_labels_train = pd.read_feather('processed-final/HarvardXPH278x2013_Spring_labels_train.feather')
ph278x_labels_test = pd.read_feather('processed-final/HarvardXPH278x2013_Spring_labels_test.feather')


# In[92]:


features_train, features_test = ph278x_features_train.drop('index', axis=1), ph278x_features_test.drop('index', axis=1)
labels_train, labels_test = ph278x_labels_train.drop('index', axis=1), ph278x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[104]:


pgrid = {'min_samples_leaf' : [50, 100],
         'min_samples_split' : [20, 30, 100, 200],
         'max_depth' : [25, 100, 200, 300],
         'n_estimators' : [500, 1000, 1500]}


# In[105]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[106]:


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


# In[107]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[108]:


grid_cv.best_estimator_


# In[109]:


grid_cv.best_score_


# In[110]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[112]:


course_metrics['course_name'].append('HarvardPH278x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 6. 6.002x (Fall) - Circuits and Electronics

# In[114]:


mit6002x_features_train = pd.read_feather('processed-final/MITx6.002x2012_Fall_features_train.feather')
mit6002x_features_test = pd.read_feather('processed-final/MITx6.002x2012_Fall_features_test.feather')
mit6002x_labels_train = pd.read_feather('processed-final/MITx6.002x2012_Fall_labels_train.feather')
mit6002x_labels_test = pd.read_feather('processed-final/MITx6.002x2012_Fall_labels_test.feather')


# In[115]:


features_train, features_test = mit6002x_features_train.drop('index', axis=1), mit6002x_features_test.drop('index', axis=1)
labels_train, labels_test = mit6002x_labels_train.drop('index', axis=1), mit6002x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[117]:


pgrid = {'min_samples_leaf' : [50, 100],
         'min_samples_split' : [20, 30, 100, 200],
         'max_depth' : [25, 100, 200, 300],
         'n_estimators' : [500, 1000, 1500]}


# In[118]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[119]:


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


# In[120]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[121]:


grid_cv.best_estimator_


# In[122]:


grid_cv.best_score_


# In[123]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[125]:


course_metrics['course_name'].append('MIT6002x_2012_Fall')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 7. 6.002x (Spring) - Circuits and Electronics

# In[127]:


mit6002x_features_train = pd.read_feather('processed-final/MITx6.002x2013_Spring_features_train.feather')
mit6002x_features_test = pd.read_feather('processed-final/MITx6.002x2013_Spring_features_test.feather')
mit6002x_labels_train = pd.read_feather('processed-final/MITx6.002x2013_Spring_labels_train.feather')
mit6002x_labels_test = pd.read_feather('processed-final/MITx6.002x2013_Spring_labels_test.feather')


# In[128]:


features_train, features_test = mit6002x_features_train.drop('index', axis=1), mit6002x_features_test.drop('index', axis=1)
labels_train, labels_test = mit6002x_labels_train.drop('index', axis=1), mit6002x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[129]:


pgrid = {'min_samples_leaf' : [50, 100],
         'min_samples_split' : [20, 30, 100, 200],
         'max_depth' : [25, 100, 200, 300],
         'n_estimators' : [500, 1000, 1500]}


# In[130]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[131]:


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


# In[132]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[133]:


grid_cv.best_estimator_


# In[134]:


grid_cv.best_score_


# In[135]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[137]:


course_metrics['course_name'].append('MIT6002x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 8. 14.73x - The Challenges of Global Poverty

# In[139]:


mit1473x_features_train = pd.read_feather('processed-final/MITx14.73x2013_Spring_features_train.feather')
mit1473x_features_test = pd.read_feather('processed-final/MITx14.73x2013_Spring_features_test.feather')
mit1473x_labels_train = pd.read_feather('processed-final/MITx14.73x2013_Spring_labels_train.feather')
mit1473x_labels_test = pd.read_feather('processed-final/MITx14.73x2013_Spring_labels_test.feather')


# In[140]:


features_train, features_test = mit1473x_features_train.drop('index', axis=1), mit1473x_features_test.drop('index', axis=1)
labels_train, labels_test = mit1473x_labels_train.drop('index', axis=1), mit1473x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[141]:


pgrid = {'min_samples_leaf' : [50, 100],
         'min_samples_split' : [20, 30, 100, 200],
         'max_depth' : [25, 100, 200, 300],
         'n_estimators' : [500, 1000, 1500]}


# In[142]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[143]:


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


# In[145]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[148]:


grid_cv.best_params_


# In[149]:


grid_cv.best_score_


# In[150]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[152]:


course_metrics['course_name'].append('MIT1473x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 9. 2.01x - Elements of Structures

# In[154]:


mit201x_features_train = pd.read_feather('processed-final/MITx2.01x2013_Spring_features_train.feather')
mit201x_features_test = pd.read_feather('processed-final/MITx2.01x2013_Spring_features_test.feather')
mit201x_labels_train = pd.read_feather('processed-final/MITx2.01x2013_Spring_labels_train.feather')
mit201x_labels_test = pd.read_feather('processed-final/MITx2.01x2013_Spring_labels_test.feather')


# In[155]:


features_train, features_test = mit201x_features_train.drop('index', axis=1), mit201x_features_test.drop('index', axis=1)
labels_train, labels_test = mit201x_labels_train.drop('index', axis=1), mit201x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[169]:


pgrid = {'min_samples_leaf' : [100, 200, 300],
         'min_samples_split' : [100, 200, 300, 500],
         'max_depth' : [200, 300, 500],
         'n_estimators' : [1500, 2000, 2500]}


# In[170]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[171]:


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


# In[172]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[173]:


grid_cv.best_params_


# In[174]:


grid_cv.best_score_


# In[150]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[152]:


course_metrics['course_name'].append('MIT201x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 10. 3.091x(Fall) - Introduction to Solid State Chemistry

# In[175]:


mit3091x_features_train = pd.read_feather('processed-final/MITx3.091x2012_Fall_features_train.feather')
mit3091x_features_test = pd.read_feather('processed-final/MITx3.091x2012_Fall_features_test.feather')
mit3091x_labels_train = pd.read_feather('processed-final/MITx3.091x2012_Fall_labels_train.feather')
mit3091x_labels_test = pd.read_feather('processed-final/MITx3.091x2012_Fall_labels_test.feather')


# In[176]:


features_train, features_test = mit3091x_features_train.drop('index', axis=1), mit3091x_features_test.drop('index', axis=1)
labels_train, labels_test = mit3091x_labels_train.drop('index', axis=1), mit3091x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[177]:


pgrid = {'min_samples_leaf' : [50, 100],
         'min_samples_split' : [20, 30, 100, 200],
         'max_depth' : [25, 100, 200, 300],
         'n_estimators' : [500, 1000, 1500]}


# In[179]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[180]:


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


# In[181]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[182]:


grid_cv.best_params_


# In[183]:


grid_cv.best_score_


# In[184]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[186]:


course_metrics['course_name'].append('MIT3091x_2012_Fall')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 11. 3.091x (Spring) - Introduction to Solid State Chemistry

# In[188]:


mit3091x_features_train = pd.read_feather('processed-final/MITx3.091x2013_Spring_features_train.feather')
mit3091x_features_test = pd.read_feather('processed-final/MITx3.091x2013_Spring_features_test.feather')
mit3091x_labels_train = pd.read_feather('processed-final/MITx3.091x2013_Spring_labels_train.feather')
mit3091x_labels_test = pd.read_feather('processed-final/MITx3.091x2013_Spring_labels_test.feather')


# In[189]:


features_train, features_test = mit3091x_features_train.drop('index', axis=1), mit3091x_features_test.drop('index', axis=1)
labels_train, labels_test = mit3091x_labels_train.drop('index', axis=1), mit3091x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[196]:


pgrid = {'min_samples_leaf' : [100, 200, 300],
         'min_samples_split' : [100, 200, 300],
         'max_depth' : [200, 300, 500],
         'n_estimators' : [1000, 1500, 2000]}


# In[197]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[198]:


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


# In[199]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[200]:


grid_cv.best_params_


# In[201]:


grid_cv.best_score_


# In[202]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[203]:


course_metrics['course_name'].append('MIT3091x_2013_Spring')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


# ### 12. 6.00x (Fall) - Introduction to Computer Science and Programming

# In[204]:


mit600x_features_train = pd.read_feather('processed-final/MITx6.00x2012_Fall_features_train.feather')
mit600x_features_test = pd.read_feather('processed-final/MITx6.00x2012_Fall_features_test.feather')
mit600x_labels_train = pd.read_feather('processed-final/MITx6.00x2012_Fall_labels_train.feather')
mit600x_labels_test = pd.read_feather('processed-final/MITx6.00x2012_Fall_labels_test.feather')


# In[205]:


features_train, features_test = mit600x_features_train.drop('index', axis=1), mit600x_features_test.drop('index', axis=1)
labels_train, labels_test = mit600x_labels_train.drop('index', axis=1), mit600x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[210]:


pgrid = {'min_samples_leaf' : [50, 100, 200],
         'min_samples_split' : [50, 100, 200],
         'max_depth' : [100, 200, 300],
         'n_estimators' : [500, 1000, 1500] }


# In[211]:


rf_classif = RandomForestClassifier(max_features='auto',
                                    random_state=20130810,
                                    n_jobs=-1)


# In[212]:


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


# In[213]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(features_train, labels_train)')


# In[214]:


grid_cv.best_params_


# In[215]:


grid_cv.best_score_


# In[216]:


F1_test_score = f1_score(grid_cv.best_estimator_.predict(features_test), labels_test)


# In[218]:


course_metrics['course_name'].append('MIT600x_2012_Fall')
course_metrics['F1_train_mean'].append(grid_cv.cv_results_['mean_test_F1'].mean())
course_metrics['F1_train_sd'].append(grid_cv.cv_results_['mean_test_F1'].std())
course_metrics['F1_test'].append(F1_test_score)


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

