
# coding: utf-8

# In[1]:


course_metrics = {'course_name' : [],
                  'val_binary_accuracy' : [],
                  'test_f1_score' : []}


# ### Design the feed-forward neural net

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K


# In[4]:


from sklearn.metrics import f1_score


# In[23]:


def build(network_type=Sequential, 
          nb_initial_layer=64,
          dense_layer_lst=[64],
          nb_final_layer=1,
          dp_rate=0.2,
          learning_rate=1e-4):
    
    model = network_type()
    
    model.add(Dense(nb_initial_layer, input_shape=(features_train.shape[1], )))
    model.add(Activation('relu'))
    
    model.add(Dropout(dp_rate))
    
    for nb_units in dense_layer_lst:
        model.add(Dense(nb_units))
        model.add(Activation('relu'))
        model.add(Dropout(dp_rate))
        
    model.add(Dense(nb_final_layer))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=[binary_accuracy])
    
    return model


# ### CB22x - Ancient Greek Hero

# In[6]:


cb22x_features_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_train.feather')
cb22x_features_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_test.feather')
cb22x_labels_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_train.feather')
cb22x_labels_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_test.feather')


# In[7]:


features_train, features_test = cb22x_features_train.drop('index', axis=1), cb22x_features_test.drop('index', axis=1)
labels_train, labels_test = cb22x_labels_train.drop('index', axis=1), cb22x_labels_test.drop('index', axis=1)


# In[8]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[9]:


features_train.shape


# In[10]:


labels_train.shape


# In[46]:


K.clear_session()


# In[47]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dp_rate=0.3, learning_rate=1e-5)
model.summary()


# In[48]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[49]:


course_metrics['course_name'].append('HarvardCB22x_2013_Spring')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### CS50x - Introduction to Compute Science I

# In[41]:


cs50x_features_train = pd.read_feather('processed-final/HarvardXCS50x2012_features_train.feather')
cs50x_features_test = pd.read_feather('processed-final//HarvardXCS50x2012_features_test.feather')
cs50x_labels_train = pd.read_feather('processed-final//HarvardXCS50x2012_labels_train.feather')
cs50x_labels_test = pd.read_feather('processed-final//HarvardXCS50x2012_labels_test.feather')


# In[42]:


features_train, features_test = cs50x_features_train.drop('index', axis=1), cs50x_features_test.drop('index', axis=1)
labels_train, labels_test = cs50x_labels_train.drop('index', axis=1), cs50x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[43]:


features_train.shape


# In[44]:


labels_train.shape


# In[64]:


K.clear_session()


# In[65]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dp_rate=0, learning_rate=1e-5)
model.summary()


# In[66]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[67]:


course_metrics['course_name'].append('HarvardCB50x_2012')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### ER22x - Justice

# In[68]:


er22x_features_train = pd.read_feather('processed-final/HarvardXER22x2013_Spring_features_train.feather')
er22x_features_test = pd.read_feather('processed-final/HarvardXER22x2013_Spring_features_test.feather')
er22x_labels_train = pd.read_feather('processed-final/HarvardXER22x2013_Spring_labels_train.feather')
er22x_labels_test = pd.read_feather('processed-final/HarvardXER22x2013_Spring_labels_test.feather')


# In[ ]:


features_train, features_test = er22x_features_train.drop('index', axis=1), er22x_features_test.drop('index', axis=1)
labels_train, labels_test = er22x_labels_train.drop('index', axis=1), er22x_labels_test.drop('index', axis=1)


# In[ ]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()

