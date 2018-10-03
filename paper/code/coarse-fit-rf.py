
# In this workbook, we fit the processed raw data to narrow down the parameter space of the Random Forests Model

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from imblearn.over_sampling import SMOTE

# Initial guesses for the parameters to tune

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


mooc_df = pd.read_feather('../data/processed-mooc-data.feather')

mooc_clean_df = pd.get_dummies(mooc_df)

features = np.array(mooc_clean_df.drop('engaged', axis=1))
labels = np.array(mooc_clean_df['engaged'])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                            test_size=0.2,
                                                                            random_state=20130810)

sm = SMOTE(random_state=20130810)
    
features_train_smote, labels_train_smote = sm.fit_sample(features_train, labels_train)

if '__name__' == '__main__':
    model_rf = RandomForestClassifier(n_jobs=-1, 
                                      warm_start=True,
                                      random_state=20130810)

    model_random_grid = RandomizedSearchCV(model_rf, 
                                           random_grid, 
                                           n_iter=50, 
                                           cv=3, 
                                           verbose=2, 
                                           random_state=20130810,
                                           n_jobs=-1)
    
    model_random_grid.fit(features_train_smote, labels_train_smote)

    pd.DataFrame(model_random_grid.best_params_).to_feather('../data/rf_best_params_random.feather')


