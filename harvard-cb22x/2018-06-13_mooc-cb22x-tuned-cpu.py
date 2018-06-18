
# coding: utf-8

# # 1. CB22x - Ancient Greek Hero

# ## Set global variables

# In[1]:


import os


# In[2]:


CONSOLIDATED_DATA_DIR = '../processed/'
COURSE_LIST = [d[0][13:] for d in os.walk(CONSOLIDATED_DATA_DIR)][1:]


# In[3]:


COURSE_LIST


# In[4]:


DATA_DIR = '../processed-final/'


# ## Design the feed-forward neural net

# In[5]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[6]:


np.random.seed(20130810)
tf.set_random_seed(20130810)


# In[7]:


import plothelpers as plh


# In[8]:


from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras.regularizers import l2

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop, Adam

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K


# In[18]:


from keras.utils import to_categorical


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# In[34]:


def build(network_type=Sequential, 
          nb_initial_layer=64,
          dense_layer_lst=[64],
          nb_final_layer=2,
          l2_penalty=0.001,
          dpt_strength=0.5,
          learning_rate=1e-4):
    
    model = network_type()
    
    model.add(Dense(nb_initial_layer, 
                    input_shape=(features_train.shape[1],),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dpt_strength))
    
    for nb_units in dense_layer_lst:
        model.add(Dense(nb_units,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(l2_penalty)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dpt_strength))
        
    model.add(Dense(nb_final_layer,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_penalty)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=[binary_accuracy])
    
    return model


# ## Tune the network

# In[11]:


course_idx = 0
print(COURSE_LIST[course_idx])


# In[12]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[24]:


features_train_df = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test_df = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train_df = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test_df = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[25]:


features_train = np.array(features_train_df)
features_test = np.array(features_test_df)

labels_train = np.array(labels_train_df).ravel()
labels_test = np.array(labels_test_df).ravel()


# In[26]:


features_train.shape


# In[27]:


labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)


# In[28]:


labels_train.shape


# In[29]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[30]:


features_train.shape, features_validation.shape


# In[31]:


labels_train.shape, labels_validation.shape


# In[35]:


K.clear_session()


# In[36]:


model = build(nb_initial_layer=32, 
              dense_layer_lst=[32, 32, 32],
              l2_penalty=0.001,
              dpt_strength=0.5,
              learning_rate=1e-3)
model.summary()


# In[37]:


# We wish to save multiple best models.
# Main purpose is to make it easier to choose the final model as we hand tune. We delete excess saved models at the end to 
# get to the best model
# This strategy would be useful if we are going to use an ensemble

out_file_path='../best-keras-runs/' +                COURSE_LIST[course_idx] +               '-8-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[ ]:


# In case you wish to save only the best model
#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[38]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=50,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[39]:


plh.plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[40]:


plh.plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ### Load the best model and compute metrics

# In[43]:


best_model = load_model('../best-keras-runs/HarvardXCB22x2013_Spring-8-16-0.65.hdf5')


# #### Training data

# In[44]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[86]:


pred_probs = best_model.predict_proba(features_train)[:, 1]


# In[90]:


plh.plot_probs(pred_probs, COURSE_LIST[course_idx], data='training')


# #### Validation data

# In[91]:


best_model.evaluate(features_validation, labels_validation, batch_size=128)


# In[92]:


pred_probs = best_model.predict_proba(features_validation)[:, 1]


# In[95]:


plh.plot_probs(pred_probs, COURSE_LIST[course_idx], data='validation')


# #### Test data

# In[ ]:


## DO NOT RUN THIS CELL TILL IT IS TIME TO REPORT RESULTS ON TEST DATA

# accuracy_score(best_model.predict_classes(features_test), labels_test)
# f1_score(best_model.predict_classes(features_test), labels_test)


# ## Explanations

# In[55]:


from lime import lime_tabular


# In[116]:


explainer = lime_tabular.LimeTabularExplainer(features_train_df, 
                                              feature_names=features_train_df.columns.tolist(),
                                              class_names=['notengaged', 'engaged'],
                                              discretize_continuous=False,
                                              verbose=True)


# ### Explore some random points

# In[128]:


i = 44
exp = explainer.explain_instance(features_train_df.iloc[i], best_model.predict_proba)


# In[193]:


exp.show_in_notebook(show_table=False, show_all=False)


# In[122]:


[feature_score[0] for feature_score in exp.as_list()[0:5]]


# ### Build a random sample of points to seek explanation

# In[141]:


from collections import Counter


# In[177]:


explainer = lime_tabular.LimeTabularExplainer(features_train_df, 
                                              feature_names=features_train_df.columns.tolist(),
                                              class_names=['notengaged', 'engaged'],
                                              discretize_continuous=False,
                                              verbose=False)


# In[187]:


nb_samples = 10 # we select a random sample 0f 5000 points from the training data to see what features were important


# In[188]:


rand_indices = np.random.choice(features_train_df.shape[0], nb_samples, replace=False)


# In[189]:


top_5 = []


# In[190]:


get_ipython().run_cell_magic('time', '', 'for idx in rand_indices:\n    exp = explainer.explain_instance(features_train_df.iloc[idx], best_model.predict_proba)\n    top_5.append([feature_score for feature_score in exp.as_list()[0:5]])')


# In[191]:


top_5


# In[184]:


best_features_list = [feature for sublist in top_5 for feature in sublist]


# In[185]:


pd.DataFrame(list(Counter(best_features_list).items()),
             columns=['Feature', 'Count']).sort_values(by='Count', ascending=False)


# In[194]:


from anchor import anchor_tabular


# In[203]:


exp = anchor_tabular.AnchorTabularExplainer(class_names=['notengaged', 'engaged'],
                                            feature_names=features_train_df.columns.tolist(),
                                            data=features_train_df)


# In[209]:


exp.fit(features_train_df, labels_train_df, features_validation_df, labels_validation_df, discretizer='decile')

