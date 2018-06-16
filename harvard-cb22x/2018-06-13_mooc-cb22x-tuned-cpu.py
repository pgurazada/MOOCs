
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


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# In[10]:


def build(network_type=Sequential, 
          nb_initial_layer=64,
          dense_layer_lst=[64],
          nb_final_layer=1,
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


# In[13]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[14]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[15]:


features_train.shape


# In[16]:


labels_train.shape


# In[17]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[18]:


features_train.shape, features_validation.shape


# In[19]:


labels_train.shape, labels_validation.shape


# In[20]:


K.clear_session()


# In[21]:


model = build(nb_initial_layer=32, 
              dense_layer_lst=[32, 32, 32],
              l2_penalty=0.001,
              dpt_strength=0.5,
              learning_rate=1e-3)
model.summary()


# In[22]:


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


# In[23]:


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


# In[24]:


plh.plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[25]:


plh.plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ### Load the best model and compute metrics

# In[26]:


best_model = load_model('HarvardXCB22x2013_Spring-2-37-0.66.hdf5')


# #### Training data

# In[27]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[28]:


pred_probs = best_model.predict_proba(features_train)


# In[29]:


pred_probs.mean()


# In[30]:


labels_train.mean()


# In[31]:


plh.plot_probs(pred_probs, COURSE_LIST[course_idx], data='training')


# #### Validation data

# In[32]:


best_model.evaluate(features_validation, labels_validation, batch_size=128)


# In[33]:


pred_probs = best_model.predict_proba(features_validation)


# In[34]:


pred_probs.mean()


# In[35]:


labels_validation.mean()


# In[36]:


plh.plot_probs(pred_probs, COURSE_LIST[course_idx], data='validation')


# #### Test data

# In[ ]:


## DO NOT RUN THIS CELL TILL IT IS TIME TO REPORT RESULTS ON TEST DATA

# accuracy_score(best_model.predict_classes(features_test), labels_test)
# f1_score(best_model.predict_classes(features_test), labels_test)

