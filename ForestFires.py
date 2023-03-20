#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


# In[14]:


df = pd.read_csv('forestfires.csv')
df.head(10)


# In[15]:


df['size_category'] = np.where(df['area']>6, '1', '0')
df['size_category']= pd.to_numeric(df['size_category'])
df.tail(10)


# In[16]:


df['day'] = ((df['day'] == 'sun') | (df['day'] == 'sat'))


# In[17]:


df = df.rename(columns = {'day' : 'is_weekend'})


# In[18]:


sns.countplot(df['is_weekend'])
plt.title('Count plot of weekend vs weekday')


# In[19]:


df.loc[:, ['rain', 'area']] = df.loc[:, ['rain', 'area']].apply(lambda x: np.log(x + 1), axis = 1)


# In[20]:


fig, ax = plt.subplots(2, figsize = (5, 8))
ax[0].hist(df['rain'])
ax[0].title.set_text('histogram of rain')
ax[1].hist(df['area'])
ax[1].title.set_text('histogram of area')


# In[29]:


features = df.drop(['size_category'], axis = 1)
labels = df['size_category'].values.reshape(-1, 1)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)


# In[31]:


sc_features = StandardScaler()


# In[46]:


history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 10, epochs = 100)


# In[33]:


model = Sequential()


# In[34]:


model.add(Dense(6, input_dim=13, activation='relu'))


# In[35]:


model.add(Dense(6, activation='relu'))


# In[36]:


model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))


# In[37]:


model.summary()


# In[38]:


model.compile(optimizer = 'adam', metrics=['accuracy'], loss ='binary_crossentropy')


# In[48]:


plt.figure(figsize=[8,5])
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.legend()
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves Epoch 100, Batch Size 10', fontsize=16)
plt.show()


# In[49]:


def fit_model(X_train, y_train, X_test, y_test, n_batch):
    model = Sequential()
model.add(Dense(6, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))


# In[50]:


model.compile(optimizer = 'adam',
metrics=['accuracy'],
loss = 'binary_crossentropy')


# In[52]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0, batch_size=n_batch)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('batch='+str(n_batch))
plt.legend()


# In[53]:


batch_sizes = [4, 6, 10, 16, 32, 64, 128, 260]
plt.figure(figsize=(10,15))
for i in range(len(batch_sizes)):


# In[54]:


plot_no = 420 + (i+1)
plt.subplot(plot_no)


# In[55]:


fit_model(X_train, y_train, X_test, y_test, batch_sizes[i])


# In[56]:


plt.show()


# In[57]:


def fit_model(trainX, trainy, validX, validy, n_epoch):


# In[58]:


model = Sequential()
model.add(Dense(6, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))


# In[59]:


model.compile(optimizer ='adam', metrics=['accuracy'], loss = 'binary_crossentropy')


# In[60]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epoch, verbose=0, batch_size=6)

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('epoch='+str(n_epoch))
plt.legend()


# In[62]:


epochs = [20, 50, 100, 120, 150, 200, 300, 400]
plt.figure(figsize=(10,15))
for i in range(len(batch_sizes)):

plot_no = 420 + (i+1)
plt.subplot(plot_no)

fit_model(X_train, y_train, X_test, y_test, epochs[i])

plt.show()


# In[65]:


model = Sequential()
model.add(Dense(6, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer ='adam',
metrics=['accuracy'],
loss = 'binary_crossentropy')


# In[64]:


return model


# In[66]:


model = init_model()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=250, verbose=0, batch_size=6, callbacks=[es, mc])


# In[67]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.legend()
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Curves', fontsize=16)
plt.show()


# In[68]:


plt.figure(figsize=[8,5])
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Valid')
plt.legend()
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()


# In[69]:


_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, valid_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Valid: %.3f' % (train_acc, valid_acc))


# In[ ]:




