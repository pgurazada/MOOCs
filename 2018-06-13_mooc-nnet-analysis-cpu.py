
# coding: utf-8

# In[ ]:


course_metrics = {'course_name' : [],
                  'val_binary_accuracy' : [],
                  'test_f1_score' : []}


# In[ ]:


import os


# In[ ]:


CONSOLIDATED_DATA_DIR = 'processed/'
COURSE_DIR_LIST = [d[0][10:] for d in os.walk(CONSOLIDATED_DATA_DIR)][1:]


# In[ ]:


COURSE_DIR_LIST


# In[ ]:


DATA_DIR = 'processed-final/'


# ### Design the feed-forward neural net

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


def build(network_type=Sequential, 
          nb_initial_layer=64,
          dense_layer_lst=[64],
          nb_final_layer=1,
          dpt_rate=0.2,
          learning_rate=1e-4):
    
    model = network_type()
    
    model.add(Dense(nb_initial_layer, input_shape=(features_train.shape[1], )))
    model.add(Activation('relu'))
    
    model.add(Dropout(dpt_rate))
    
    for nb_units in dense_layer_lst:
        model.add(Dense(nb_units))
        model.add(Activation('relu'))
        model.add(Dropout(dpt_rate))
        
    model.add(Dense(nb_final_layer))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=[binary_accuracy])
    
    return model


# ### 1. CB22x - Ancient Greek Hero

# In[ ]:


cb22x_features_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_train.feather')
cb22x_features_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_features_test.feather')
cb22x_labels_train = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_train.feather')
cb22x_labels_test = pd.read_feather('processed-final/HarvardXCB22x2013_Spring_labels_test.feather')


# In[ ]:


features_train, features_test = cb22x_features_train.drop('index', axis=1), cb22x_features_test.drop('index', axis=1)
labels_train, labels_test = cb22x_labels_train.drop('index', axis=1), cb22x_labels_test.drop('index', axis=1)


# In[ ]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[ ]:


features_train.shape


# In[ ]:


labels_train.shape


# In[ ]:


K.clear_session()


# In[ ]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.2, learning_rate=1e-5)
model.summary()


# In[ ]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[ ]:


course_metrics['course_name'].append('HarvardCB22x_2013_Spring')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 2. CS50x - Introduction to Computer Science I

# In[ ]:


cs50x_features_train = pd.read_feather('processed-final/HarvardXCS50x2012_features_train.feather')
cs50x_features_test = pd.read_feather('processed-final//HarvardXCS50x2012_features_test.feather')
cs50x_labels_train = pd.read_feather('processed-final//HarvardXCS50x2012_labels_train.feather')
cs50x_labels_test = pd.read_feather('processed-final//HarvardXCS50x2012_labels_test.feather')


# In[ ]:


features_train, features_test = cs50x_features_train.drop('index', axis=1), cs50x_features_test.drop('index', axis=1)
labels_train, labels_test = cs50x_labels_train.drop('index', axis=1), cs50x_labels_test.drop('index', axis=1)

features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[ ]:


features_train.shape


# In[ ]:


labels_train.shape


# In[ ]:


K.clear_session()


# In[ ]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0, learning_rate=1e-5)
model.summary()


# In[ ]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[ ]:


course_metrics['course_name'].append('HarvardCB50x_2012')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 3. ER22x - Justice

# In[ ]:


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


# In[ ]:


features_train.shape


# In[ ]:


labels_train.shape


# In[ ]:


K.clear_session()


# In[ ]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.1, learning_rate=1e-5)
model.summary()


# In[ ]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[ ]:


course_metrics['course_name'].append('HarvardXER22x2013_Spring')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 4. PH207x - Health in Numbers: Quantitative Methods in Clinical & Public Health Research

# In[ ]:


ph207x_features_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_train.feather')
ph207x_features_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_test.feather')
ph207x_labels_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_train.feather')
ph207x_labels_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_test.feather')


# In[ ]:


features_train, features_test = ph207x_features_train.drop('index', axis=1), ph207x_features_test.drop('index', axis=1)
labels_train, labels_test = ph207x_labels_train.drop('index', axis=1), ph207x_labels_test.drop('index', axis=1)


# In[ ]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[ ]:


features_train.shape


# In[ ]:


labels_train.shape


# In[ ]:


K.clear_session()


# In[ ]:


model = build(nb_initial_layer=64, dense_layer_lst=[32, 32, 32], dpt_rate=0.05, learning_rate=1e-2)
model.summary()


# In[ ]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=50,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[ ]:


course_metrics['course_name'].append('HarvardXER22x2013_Spring')
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 5. PH278x - Human Health and Global Environmental Change

# In[ ]:


course_id = 4


# In[ ]:


features_train = pd.read_feather(SPLIT_DATA_DIR + COURSE_DIR_LIST[course_id] + '_features_train.feather')


# In[ ]:


features_train.shape


# In[ ]:


ph207x_features_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_train.feather')
ph207x_features_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_features_test.feather')
ph207x_labels_train = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_train.feather')
ph207x_labels_test = pd.read_feather('processed-final/HarvardXPH207x2012_Fall_labels_test.feather')


# In[ ]:


features_train, features_test = ph207x_features_train.drop('index', axis=1), ph207x_features_test.drop('index', axis=1)
labels_train, labels_test = ph207x_labels_train.drop('index', axis=1), ph207x_labels_test.drop('index', axis=1)


# In[ ]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[ ]:


features_train.shape


# In[ ]:


labels_train.shape

