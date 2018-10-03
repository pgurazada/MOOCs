
# coding: utf-8

# ## Set global variables

# In[1]:


course_metrics = {'course_name' : [],
                  'val_binary_accuracy' : [],
                  'test_accuracy' : [],
                  'test_f1_score' : [] }


# In[2]:


import os


# In[3]:


CONSOLIDATED_DATA_DIR = 'processed/'
COURSE_LIST = [d[0][10:] for d in os.walk(CONSOLIDATED_DATA_DIR)][1:]


# In[4]:


COURSE_LIST


# In[5]:


DATA_DIR = 'processed-final/'


# ## Design the feed-forward neural net

# In[6]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[7]:


np.random.seed(20130810)
tf.set_random_seed(20130810)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


get_ipython().magic('matplotlib inline')
sns.set_style('ticks', {'grid_color' : '0.9'})
sns.set_context('talk', font_scale=1.2)
sns.set_palette('gray')


# In[10]:


from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Dropout

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop, Adam

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# In[12]:


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
    
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=binary_crossentropy,
                  metrics=[binary_accuracy])
    
    return model


# In[13]:


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


# In[14]:


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


# ## Tune the network for each course

# ### 1. CB22x - Ancient Greek Hero

# In[15]:


course_idx = 0
print(COURSE_LIST[course_idx])


# In[16]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[17]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[18]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[19]:


features_train.shape


# In[20]:


labels_train.shape


# In[21]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[22]:


features_train.shape, features_validation.shape


# In[23]:


labels_train.shape, labels_validation.shape


# In[24]:


K.clear_session()


# In[25]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.2, learning_rate=1e-3)
model.summary()


# In[27]:


# We wish to save multiple best models.
# Main purpose is to make it easier to choose the final model as we hand tune. We delete excess saved models at the end to 
# get to the best model
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[ ]:


# In case you wish to save only the best model
#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[28]:


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


# In[29]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[30]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[31]:


best_model = load_model('best-keras-runs/HarvardXCB22x2013_Spring-24-0.65.hdf5')


# In[32]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[33]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_train, labels_train, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 2. CS50x - Introduction to Computer Science I

# In[34]:


course_idx = 1
print(COURSE_LIST[course_idx])


# In[35]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[36]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[37]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[38]:


features_train.shape


# In[39]:


labels_train.shape


# In[40]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[41]:


features_train.shape, features_validation.shape


# In[42]:


labels_train.shape, labels_validation.shape


# In[43]:


K.clear_session()


# In[44]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.1, learning_rate=1e-4)
model.summary()


# In[45]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[ ]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[46]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[47]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[48]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[49]:


best_model = load_model('best-keras-runs/HarvardXCS50x2012-33-0.68.hdf5')


# In[50]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[51]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_train, labels_train, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 3. ER22x - Justice

# In[52]:


course_idx = 2
print(COURSE_LIST[course_idx])


# In[53]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[54]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[55]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[56]:


features_train.shape


# In[57]:


labels_train.shape


# In[58]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[59]:


features_train.shape, features_validation.shape


# In[60]:


labels_train.shape, labels_validation.shape


# In[71]:


K.clear_session()


# In[72]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.1, learning_rate=1e-5)
model.summary()


# In[69]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[ ]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[73]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[74]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[75]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[76]:


best_model = load_model('best-keras-runs/HarvardXER22x2013_Spring-18-0.67.hdf5')


# In[77]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[78]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_train, labels_train, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 4. PH207x - Health in Numbers: Quantitative Methods in Clinical & Public Health Research

# In[79]:


course_idx = 3
print(COURSE_LIST[course_idx])


# In[80]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[81]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[82]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[83]:


features_train.shape


# In[84]:


labels_train.shape


# In[85]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[86]:


features_train.shape, features_validation.shape


# In[87]:


labels_train.shape, labels_validation.shape


# In[132]:


K.clear_session()


# In[133]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.01, learning_rate=1e-4)
model.summary()


# In[135]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-8-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[130]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[136]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[137]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[138]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[139]:


best_model = load_model('best-keras-runs/HarvardXPH207x2012_Fall-14-0.65.hdf5')


# In[140]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[141]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_train, labels_train, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 5. PH278x - Human Health and Global Environmental Change

# In[142]:


course_idx = 4
print(COURSE_LIST[course_idx])


# In[143]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[144]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[145]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[146]:


features_train.shape


# In[147]:


labels_train.shape


# In[148]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[149]:


features_train.shape, features_validation.shape


# In[150]:


labels_train.shape, labels_validation.shape


# In[180]:


K.clear_session()


# In[181]:


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.2, learning_rate=1e-4)
model.summary()


# In[182]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-6-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[130]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[183]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[184]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[185]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[186]:


best_model = load_model('best-keras-runs/HarvardXPH278x2013_Spring-4-07-0.72.hdf5')


# In[187]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[197]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_train, labels_train, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 6. MIT 14.73x - The Challenges of Global Poverty 

# In[198]:


course_idx = 5
print(COURSE_LIST[course_idx])


# In[199]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[200]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[201]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[202]:


features_train.shape


# In[203]:


labels_train.shape


# In[204]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[205]:


features_train.shape, features_validation.shape


# In[206]:


labels_train.shape, labels_validation.shape


# In[246]:


K.clear_session()


# In[247]:


model = build(nb_initial_layer=64, dense_layer_lst=[32, 32, 32], dpt_rate=0.05, learning_rate=1e-4)
model.summary()


# In[248]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-8-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[130]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[249]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[250]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[251]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[252]:


best_model = load_model('best-keras-runs/MITx14.73x2013_Spring-8-37-0.63.hdf5')


# In[258]:


best_model.evaluate(features_validation, labels_validation, batch_size=64)


# In[259]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_validation, labels_validation, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 7. MIT 2.01x - Elements of Structures

# In[301]:


course_idx = 6
print(COURSE_LIST[course_idx])


# In[302]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[303]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[304]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[305]:


features_train.shape


# In[306]:


labels_train.shape


# In[307]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[308]:


features_train.shape, features_validation.shape


# In[309]:


labels_train.shape, labels_validation.shape


# In[337]:


K.clear_session()


# In[338]:


model = build(nb_initial_layer=64, dense_layer_lst=[32, 32, 32], dpt_rate=0.05, learning_rate=1e-5)
model.summary()


# In[339]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-11-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[130]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[340]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[341]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[342]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[343]:


best_model = load_model('best-keras-runs/MITx2.01x2013_Spring-11-26-0.66.hdf5')


# In[344]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[345]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_validation, labels_validation, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 8. MIT 3.091x (Fall) - Introduction to Solid State Chemistry

# In[346]:


course_idx = 7
print(COURSE_LIST[course_idx])


# In[347]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[348]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


# In[349]:


features_train = np.array(features_train)
features_test = np.array(features_test)

labels_train = np.array(labels_train).ravel()
labels_test = np.array(labels_test).ravel()


# In[350]:


features_train.shape


# In[351]:


labels_train.shape


# In[352]:


features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, 
                                                                                        test_size=0.2, 
                                                                                        random_state=20130810)


# In[353]:


features_train.shape, features_validation.shape


# In[354]:


labels_train.shape, labels_validation.shape


# In[382]:


K.clear_session()


# In[383]:


model = build(nb_initial_layer=64, dense_layer_lst=[32, 32, 32], dpt_rate=0.01, learning_rate=1e-5)
model.summary()


# In[384]:


# We wish to save multiple best models.
# This strategy would be useful if we are going to use an ensemble

out_file_path='best-keras-runs/' +                COURSE_LIST[course_idx] +               '-6-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'


# In[130]:


#out_file_path='best-keras-runs/' + \
#               COURSE_LIST[course_idx] + \
#              '-best-model.hdf5'


# In[385]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=100,
                         validation_data=[features_validation, labels_validation],
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6),
                                    ModelCheckpoint(out_file_path, 
                                                    monitor='val_binary_accuracy',
                                                    mode='max',
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    save_weights_only=False)])


# In[386]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[387]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# ##### Load the best model and compute metrics

# In[343]:


best_model = load_model('best-keras-runs/MITx2.01x2013_Spring-11-26-0.66.hdf5')


# In[344]:


best_model.evaluate(features_train, labels_train, batch_size=128)


# In[345]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(best_model.evaluate(features_validation, labels_validation, batch_size=128)[-1])

course_metrics['test_accuracy'].append(accuracy_score(best_model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(best_model.predict_classes(features_test), labels_test))


# ### 9. MIT 3.091x (Spring) - Introduction to Solid State Chemistry 

# In[ ]:


course_idx = 8
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 10. MIT 6.002x (Fall) - Circuits and Electronics

# In[ ]:


course_idx = 9
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 11. MIT 6.002x (Spring) - Circuits and Electronics

# In[ ]:


course_idx = 10
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 12. MIT 6.00x (Fall) - Introduction to Computer Science

# In[ ]:


course_idx = 11
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 13. MIT 6.00x (Spring) - Introduction to Computer Science

# In[ ]:


course_idx = 12
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 14. MIT 7.00x - Introduction to Biology - secret of life

# In[ ]:


course_idx = 13
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 15. MIT 8.02x - Electricity and Magnetism

# In[ ]:


course_idx = 14
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 16. MIT 8.MReV - Mechanics Review

# In[ ]:


course_idx = 15
print(COURSE_LIST[course_idx])


# In[ ]:


course_loc = DATA_DIR + COURSE_LIST[course_idx]
print(course_loc)


# In[ ]:


features_train = pd.read_feather(course_loc + '_features_train.feather').drop('index', axis=1)
features_test = pd.read_feather(course_loc + '_features_test.feather').drop('index', axis=1)

labels_train = pd.read_feather(course_loc + '_labels_train.feather').drop('index', axis=1)
labels_test = pd.read_feather(course_loc + '_labels_test.feather').drop('index', axis=1)


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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))

