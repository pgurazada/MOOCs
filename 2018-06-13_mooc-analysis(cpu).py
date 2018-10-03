
# coding: utf-8

# In[45]:


course_metrics = {'course_name' : [],
                  'val_binary_accuracy' : [],
                  'test_f1_score' : []}


# ### Design the MLP

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier

from keras import backend as K


# In[3]:


from sklearn.metrics import f1_score


# ### Assembling the neural net for the largest data set

# In[4]:


cs50x_features_train = pd.read_feather('processed-final/HarvardXCS50x2012_features_train.feather')
cs50x_features_test = pd.read_feather('processed-final//HarvardXCS50x2012_features_test.feather')
cs50x_labels_train = pd.read_feather('processed-final//HarvardXCS50x2012_labels_train.feather')
cs50x_labels_test = pd.read_feather('processed-final//HarvardXCS50x2012_labels_test.feather')


# In[5]:


features_train, features_test = cs50x_features_train.drop('index', axis=1), cs50x_features_test.drop('index', axis=1)
labels_train, labels_test = cs50x_labels_train.drop('index', axis=1), cs50x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[6]:


features_train.shape


# In[7]:


labels_train.shape


# In[9]:


def build(network_type=Sequential, 
          nb_initial_layer=64,
          dense_layer_lst=[64],
          nb_final_layer=1,
          learning_rate=1e-4):
    
    model = network_type()
    
    model.add(Dense(nb_initial_layer, input_shape=(features_train.shape[1], )))
    model.add(Activation('relu'))
    
    for nb_units in dense_layer_lst:
        model.add(Dense(nb_units))
        model.add(Activation('relu'))
        
    model.add(Dense(nb_final_layer))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=[binary_accuracy])
    
    return model


# In[33]:


K.clear_session()


# In[34]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], learning_rate=1e-5)
model.summary()


# In[35]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[44]:


model_output.history['val_binary_accuracy'][-1]


# In[38]:


f1_score(model.predict_classes(features_test), labels_test)

