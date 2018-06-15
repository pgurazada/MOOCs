
# coding: utf-8

# # 1. CB22x - Ancient Greek Hero

# ## Set global variables

# In[1]:


import os


# In[6]:


CONSOLIDATED_DATA_DIR = '../processed/'
COURSE_LIST = [d[0][13:] for d in os.walk(CONSOLIDATED_DATA_DIR)][1:]


# In[7]:


COURSE_LIST


# In[8]:


DATA_DIR = '../processed-final/'


# ## Design the feed-forward neural net

# In[9]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[10]:


np.random.seed(20130810)
tf.set_random_seed(20130810)


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


get_ipython().magic('matplotlib inline')

sns.set_context('talk', font_scale=1.2)
sns.set_palette('gray')
sns.set_style('ticks', {'grid_color' : 0.6})


# In[24]:


from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras.regularizers import l2

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop, Adam

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# In[ ]:


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


# In[ ]:


def plot_loss(fit_history, course_name):
    epochs = range(1, len(fit_history['binary_accuracy'])+1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, fit_history['loss'], '--', label='Training loss')
    plt.plot(epochs, fit_history['val_loss'], '-', label='Validation loss')
    
    plt.title('Training and Validation loss for ' + course_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()


# In[ ]:


def plot_accuracy(fit_history, course_name):
    epochs = range(1, len(fit_history['binary_accuracy'])+1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, fit_history['binary_accuracy'], '--', label='Training Accuracy')
    plt.plot(epochs, fit_history['val_binary_accuracy'], '-', label='Validation Accuracy')
    
    plt.title('Training and Validation accuracy for ' + course_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()


# ## Tune the network

# In[13]:


course_idx = 0
print(COURSE_LIST[course_idx])


# In[14]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[15]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[16]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[17]:


features_train.shape


# In[18]:


labels_train.shape


# In[20]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[21]:


features_train.shape, features_validation.shape


# In[22]:


labels_train.shape, labels_validation.shape


# In[ ]:


K.clear_session()


# In[ ]:


model = build(nb_initial_layer=32, 
              dense_layer_lst=[32, 32, 32],
              l2_penalty=0.001,
              dpt_strength=0.5,
              learning_rate=1e-3)
model.summary()


# In[ ]:


# We wish to save multiple best models.
# Main purpose is to make it easier to choose the final model as we hand tune. We delete excess saved models at the end to 
# get to the best model
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-8-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[ ]:


# In case you wish to save only the best model
#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[ ]:


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


# In[ ]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[ ]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ### Load the best model and compute metrics

# In[25]:


best_model = load_model('HarvardXCB22x2013_Spring-2-37-0.66.hdf5')


# In[26]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[27]:


pred_probs = best_model.predict_proba(features_train)


# In[28]:


pred_probs.mean()


# In[29]:


labels_train.mean()


# In[30]:


plt.figure(figsize=(12,6))
plt.hist(pred_probs)
plt.xlabel('Predicted probability')
plt.ylabel('Count')
plt.title('Distribution of predicted probabilities on the training data')
plt.show()


# In[ ]:


## DO NOT RUN THIS CELL TILL IT IS TIME TO REPORT RESULTS ON TEST DATA

# accuracy_score(best_model.predict_classes(features_test), labels_test)
# f1_score(best_model.predict_classes(features_test), labels_test)

