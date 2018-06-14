
# coding: utf-8

# ## Set global variables

# In[ ]:


course_metrics = {'course_name' : [],
                  'val_binary_accuracy' : [],
                  'test_accuracy' : [],
                  'test_f1_score' : [] }


# In[ ]:


import os


# In[ ]:


CONSOLIDATED_DATA_DIR = 'processed/'
COURSE_LIST = [d[0][10:] for d in os.walk(CONSOLIDATED_DATA_DIR)][1:]


# In[ ]:


COURSE_LIST


# In[ ]:


DATA_DIR = 'processed-final/'


# ## Design the feed-forward neural net

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


np.random.seed(20130810)
tf.set_random_seed(20130810)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().magic('matplotlib inline')
sns.set_style('ticks', {'grid_color' : '0.9'})
sns.set_context('talk', font_scale=1.2)
sns.set_palette('gray')


# In[ ]:


from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.losses import binary_crossentropy

from keras.optimizers import RMSprop

from keras.metrics import binary_accuracy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score


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


# ## Tune the network for each course

# ### 1. CB22x - Ancient Greek Hero

# In[ ]:


course_idx = 0
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


model = build(nb_initial_layer=32, dense_layer_lst=[32, 32, 32], dpt_rate=0.2, learning_rate=1e-5)
model.summary()


# In[ ]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=50,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), 
                                    ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[ ]:


plot_loss(model_output.history, COURSE_LIST[course_idx])


# In[ ]:


plot_accuracy(model_output.history, COURSE_LIST[course_idx])


# In[ ]:


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 2. CS50x - Introduction to Computer Science I

# In[ ]:


course_idx = 1
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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 3. ER22x - Justice

# In[ ]:


course_idx = 2
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


course_metrics['course_name'].append(COURSE_LIST[course_idx])
course_metrics['val_binary_accuracy'].append(model_output.history['val_binary_accuracy'][-1])
course_metrics['test_accuracy'].append(accuracy_score(model.predict_classes(features_test), labels_test))
course_metrics['test_f1_score'].append(f1_score(model.predict_classes(features_test), labels_test))


# ### 4. PH207x - Health in Numbers: Quantitative Methods in Clinical & Public Health Research

# In[ ]:


course_idx = 3
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


# ### 5. PH278x - Human Health and Global Environmental Change

# In[ ]:


course_idx = 4
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


# ### 6. MIT 14.73x - The Challenges of Global Poverty 

# In[ ]:


course_idx = 5
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


# ### 7. MIT 2.01x - Elements of Structures

# In[ ]:


course_idx = 6
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


# ### 8. MIT 3.091x (Fall) - Introduction to Solid State Chemistry

# In[ ]:


course_idx = 7
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

