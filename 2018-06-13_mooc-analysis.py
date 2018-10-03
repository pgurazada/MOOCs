
# coding: utf-8

# [View in Colaboratory](https://colab.research.google.com/github/pgurazada/ml-projects/blob/master/2018-06-13_mooc-analysis.ipynb)

# ### Connect Google Drive

# In[ ]:


get_ipython().system('apt-get install -y -qq software-properties-common python-software-properties module-init-tools')
get_ipython().system('add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null')
get_ipython().system('apt-get update -qq 2>&1 > /dev/null')
get_ipython().system('apt-get -y install -qq google-drive-ocamlfuse fuse')


# In[ ]:


from google.colab import auth
auth.authenticate_user()


# In[ ]:


from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()


# In[ ]:


import getpass


# In[ ]:


get_ipython().system('google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL')
vcode = getpass.getpass()
get_ipython().system('echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}')


# In[ ]:


get_ipython().system('mkdir -p drive')
get_ipython().system('google-drive-ocamlfuse drive')


# In[ ]:


get_ipython().system('ls drive/data/moocs')


# ### Design the MLP

# In[ ]:


get_ipython().system('pip install -U feather-format')


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


# In[54]:


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


# In[8]:


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
  
  model.compile(optimizer=RMSprop(lr=learning_rate),
                loss=binary_crossentropy,
                metrics=[binary_accuracy])
  
  return model


# In[51]:


K.clear_session()


# In[52]:


model = build(dense_layer_lst=[32, 32, 32])
model.summary()


# In[53]:


model_output = model.fit(features_train, labels_train,
                         batch_size=128,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[EarlyStopping(patience=4), ReduceLROnPlateau(patience=4, min_lr=1e-6)])


# In[55]:


f1_score(model.predict(features_test), labels_test)

