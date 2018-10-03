import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV

from imblearn.over_sampling import SMOTE

def fit_model(mooc_df, classifier_modelobj):
    
    data_clean_df = pd.get_dummies(mooc_df)

    labels = np.array(data_clean_df.engaged)

    features = np.array(data_clean_df.drop(engaged, axis=1))




