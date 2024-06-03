#!/usr/bin/env python
# coding: utf-8

# # Section 3.1

# In[ ]:


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy as np
from keras.models import model_from_json
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
import os
import warnings
import seaborn as sn
from sklearn.decomposition import PCA
import os
import pickle
import matplotlib.pylab as plt
import matplotlib.transforms
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import math
import time


# ### Loading datasets fro training and testing

# In[2]:


# Training feature sets
feature1 = pd.read_csv("subset_1/dataset_subset_1.csv")
feature2 = pd.read_csv("subset_2/dataset_subset_2.csv")
feature3 = pd.read_csv("subset_3/dataset_subset_3.csv")
feature4 = pd.read_csv("subset_4/dataset_subset_4.csv")


# In[3]:


# Testing feature sets
featureTest1 = pd.read_csv("subset_1/dataset_validation_1.csv")
featureTest2 = pd.read_csv("subset_2/dataset_validation_2.csv")
featureTest3 = pd.read_csv("subset_3/dataset_validation_3.csv")
featureTest4 = pd.read_csv("subset_4/dataset_validation_4.csv")


# In[4]:


# Training labels
adr1 = pd.read_csv("subset_1/ADR_dataset_for_training_subset_1.csv")
adr2 = pd.read_csv("subset_2/ADR_dataset_for_training_subset_2.csv")
adr3 = pd.read_csv("subset_3/ADR_dataset_for_training_subset_3.csv")
adr4 = pd.read_csv("subset_4/ADR_dataset_for_training_subset_4.csv")


# In[5]:


# Testing labels
adrTest1 = pd.read_csv("subset_1/ADR_validation_for_validation_subset_1.csv")
adrTest2 = pd.read_csv("subset_2/ADR_validation_for_validation_subset_2.csv")
adrTest3 = pd.read_csv("subset_3/ADR_validation_for_validation_subset_3.csv")
adrTest4 = pd.read_csv("subset_4/ADR_validation_for_validation_subset_4.csv")


# In[6]:


adr1.drop("Unnamed: 0", axis=1, inplace=True)
adr2.drop("Unnamed: 0", axis=1, inplace=True)
adr3.drop("Unnamed: 0", axis=1, inplace=True)
adr4.drop("Unnamed: 0", axis=1, inplace=True)

adrTest1.drop("Unnamed: 0", axis=1, inplace=True)
adrTest2.drop("Unnamed: 0", axis=1, inplace=True)
adrTest3.drop("Unnamed: 0", axis=1, inplace=True)
adrTest4.drop("Unnamed: 0", axis=1, inplace=True)

feature1.drop("Unnamed: 0", axis=1, inplace=True)
feature2.drop("Unnamed: 0", axis=1, inplace=True)
feature3.drop("Unnamed: 0", axis=1, inplace=True)
feature4.drop("Unnamed: 0", axis=1, inplace=True)

featureTest1.drop("Unnamed: 0", axis=1, inplace=True)
featureTest2.drop("Unnamed: 0", axis=1, inplace=True)
featureTest3.drop("Unnamed: 0", axis=1, inplace=True)
featureTest4.drop("Unnamed: 0", axis=1, inplace=True)


# In[7]:


feature1.drop("combined_pert", axis=1, inplace=True)
feature2.drop("combined_pert", axis=1, inplace=True)
feature3.drop("combined_pert", axis=1, inplace=True)
feature4.drop("combined_pert", axis=1, inplace=True)

featureTest1.drop("combined_pert", axis=1, inplace=True)
featureTest2.drop("combined_pert", axis=1, inplace=True)
featureTest3.drop("combined_pert", axis=1, inplace=True)
featureTest4.drop("combined_pert", axis=1, inplace=True)


# ### 50% Data

# In[44]:


div = pd.read_excel("fifty_percent.xlsx")
adr_fifty = div['ADR']


# In[50]:


adr1_fifty = adr1[adr_fifty]
adr2_fifty = adr2[adr_fifty]
adr3_fifty = adr3[adr_fifty]
adr4_fifty = adr4[adr_fifty]

adrTest1_fifty = adrTest1[adr_fifty]
adrTest2_fifty = adrTest2[adr_fifty]
adrTest3_fifty = adrTest3[adr_fifty]
adrTest4_fifty = adrTest4[adr_fifty]


# In[ ]:


features = [feature1, feature2, feature3, feature4]
featureTests = [featureTest1, featureTest2, featureTest3, featureTest4]
adrs = [adr1_fifty, adr2_fifty, adr3_fifty, adr4_fifty]
adrTests = [adrTest1_fifty, adrTest2_fifty, adrTest3_fifty, adrTest4_fifty]
models = {"subset": [],
          "patience": [],
          "trainingAcc": [],
          "testingAcc": [],
          "trainAuc": [],
          "testAuc": [],
          "trainPre": [],
          "testPre": [],
          "trainRe": [],
          "testRe": [],
          "trainingLoss": [],
          "testingLoss": [],
          }


# In[55]:


len(adr1_fifty.columns)


# In[ ]:


counter = 0
for i in range(4):
    for p in [2,5,8,10,13,15,18,20]:
        # Preparing model
        model = Sequential()
        model.add(Dense(5000, input_dim=11164, activation='relu'))
        model.add(Dense(4000, activation='relu'))
        model.add(Dense(3000, activation='relu'))
        model.add(Dense(68, activation='sigmoid'))
        opt = SGD(lr=0.1, momentum=0.9)
        if counter==0:
            count = ""
        else:
            count = "_" + str(counter)
        
        #Saving Results
        es_callback = EarlyStopping(monitor='val_loss', patience=p)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(features[i], adrs[i] , validation_data=(featureTests[i], adrTests[i]), epochs=200, batch_size=10000, verbose=1, callbacks=[es_callback])
        models["trainingAcc"].append(np.mean(history.history['accuracy']))
        models["testingAcc"].append(np.mean(history.history['val_accuracy']))
        models["trainingLoss"].append(np.mean(history.history['loss']))
        models["testingLoss"].append(np.mean(history.history['val_loss']))
        models["trainAuc"].append(np.mean(history.history['auc'+str(counter)]))
        models["testAuc"].append(np.mean(history.history['val_auc'+str(counter)]))
        models["trainPre"].append(np.mean(history.history['precision'+str(counter)]))
        models["testPre"].append(np.mean(history.history['val_precision'+str(counter)]))
        models["trainRe"].append(np.mean(history.history['recall'+str(counter)]))
        models["testRe"].append(np.mean(history.history['val_recall'+str(counter)]))
        models["patience"].append(p)
        models["subset"].append(i)
        
        #saving model
        model_json = model.to_json()
        with open("subset_" + str(i+1) + "/model-fifty-" + str(i) + "-" + str(p) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("subset_" + str(i+1) + "/model-fifty" + str(i) + "-" + str(p) + ".h5")
        print("Saved model to disk")
        counter += 1


# In[ ]:


df = pd.DataFrame.from_dict(models)
df


# In[ ]:


df.to_csv("fifty.csv")


# ### 30% Data

# In[ ]:


div = pd.read_excel("fifty_percent.xlsx")
adr_thirty = div['ADR']


# In[ ]:


adr1_thirty = adr1[adr_fifty]
adr2_thirty = adr2[adr_fifty]
adr3_thirty = adr3[adr_fifty]
adr4_thirty = adr4[adr_fifty]

adrTest1_thirty = adrTest1[adr_fifty]
adrTest2_thirty = adrTest2[adr_fifty]
adrTest3_thirty = adrTest3[adr_fifty]
adrTest4_thirty = adrTest4[adr_fifty]


# In[ ]:


features = [feature1, feature2, feature3, feature4]
featureTests = [featureTest1, featureTest2, featureTest3, featureTest4]
adrs = [adr1_thirty, adr2_thirty, adr3_thirty, adr4_thirty]
adrTests = [adrTest1_thirty, adrTest2_thirty, adrTest3_thirty, adrTest4_thirty]
models = {"subset": [],
          "patience": [],
          "trainingAcc": [],
          "testingAcc": [],
          "trainAuc": [],
          "testAuc": [],
          "trainPre": [],
          "testPre": [],
          "trainRe": [],
          "testRe": [],
          "trainingLoss": [],
          "testingLoss": [],
          }


# In[ ]:


len(adr1_thirty.columns)


# In[ ]:


counter = 0
for i in range(4):
    for p in [2,5,8,10,13,15,18,20]:
        model = Sequential()
        model.add(Dense(5000, input_dim=11164, activation='relu'))
        model.add(Dense(4000, activation='relu'))
        model.add(Dense(3000, activation='relu'))
        model.add(Dense(165, activation='sigmoid'))
        opt = SGD(lr=0.1, momentum=0.9)
        if counter==0:
            count = ""
        else:
            count = "_" + str(counter)
            
        es_callback = EarlyStopping(monitor='val_loss', patience=p)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(features[i], adrs[i] , validation_data=(featureTests[i], adrTests[i]), epochs=200, batch_size=10000, verbose=1, callbacks=[es_callback])
        models["trainingAcc"].append(np.mean(history.history['accuracy']))
        models["testingAcc"].append(np.mean(history.history['val_accuracy']))
        models["trainingLoss"].append(np.mean(history.history['loss']))
        models["testingLoss"].append(np.mean(history.history['val_loss']))
        models["trainAuc"].append(np.mean(history.history['auc'+str(counter)]))
        models["testAuc"].append(np.mean(history.history['val_auc'+str(counter)]))
        models["trainPre"].append(np.mean(history.history['precision'+str(counter)]))
        models["testPre"].append(np.mean(history.history['val_precision'+str(counter)]))
        models["trainRe"].append(np.mean(history.history['recall'+str(counter)]))
        models["testRe"].append(np.mean(history.history['val_recall'+str(counter)]))
        models["patience"].append(p)
        models["subset"].append(i)
        model_json = model.to_json()
        with open("subset_" + str(i+1) + "/model-thirty-" + str(i) + "-" + str(p) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("subset_" + str(i+1) + "/model-thirty" + str(i) + "-" + str(p) + ".h5")
        print("Saved model to disk")
        counter += 1


# In[ ]:


df = pd.DataFrame.from_dict(models)
df


# In[ ]:


df.to_csv("thirty.csv")


# ### 10% Data

# In[ ]:


features = [feature1, feature2, feature3, feature4]
featureTests = [featureTest1, featureTest2, featureTest3, featureTest4]
adrs = [adr1, adr2, adr3, adr4]
adrTests = [adrTest1, adrTest2, adrTest3, adrTest4]
models = {"subset": [],
          "patience": [],
          "trainingAcc": [],
          "testingAcc": [],
          "trainAuc": [],
          "testAuc": [],
          "trainPre": [],
          "testPre": [],
          "trainRe": [],
          "testRe": [],
          "trainingLoss": [],
          "testingLoss": [],
          }


# In[12]:


counter = 0
for i in range(4):
    for p in [2,5,8,10,13,15,18,20]:
        model = Sequential()
        model.add(Dense(5000, input_dim=11164, activation='relu'))
        model.add(Dense(4000, activation='relu'))
        model.add(Dense(3000, activation='relu'))
        model.add(Dense(243, activation='sigmoid'))
        opt = SGD(lr=0.1, momentum=0.9)
        if counter==0:
            count = ""
        else:
            count = "_" + str(counter)
            
        es_callback = EarlyStopping(monitor='val_loss', patience=p)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(features[i], adrs[i] , validation_data=(featureTests[i], adrTests[i]), epochs=200, batch_size=10000, verbose=1, callbacks=[es_callback])
        models["trainingAcc"].append(np.mean(history.history['accuracy']))
        models["testingAcc"].append(np.mean(history.history['val_accuracy']))
        models["trainingLoss"].append(np.mean(history.history['loss']))
        models["testingLoss"].append(np.mean(history.history['val_loss']))
        models["trainAuc"].append(np.mean(history.history['auc'+str(counter)]))
        models["testAuc"].append(np.mean(history.history['val_auc'+str(counter)]))
        models["trainPre"].append(np.mean(history.history['precision'+str(counter)]))
        models["testPre"].append(np.mean(history.history['val_precision'+str(counter)]))
        models["trainRe"].append(np.mean(history.history['recall'+str(counter)]))
        models["testRe"].append(np.mean(history.history['val_recall'+str(counter)]))
        models["patience"].append(p)
        models["subset"].append(i)
        model_json = model.to_json()
        with open("subset_" + str(i+1) + "/model-ten" + str(i) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("subset_" + str(i+1) + "/model-ten" + str(i) + ".h5")
        print("Saved model to disk")
        counter += 1


# In[60]:


df = pd.DataFrame.from_dict(models)
df


# In[61]:


df.to_csv("ten.csv")


# ### 90% Data

# In[6]:


n = pd.read_csv("ADR_combined_ninety.csv")
n.drop("Unnamed: 0", axis=1, inplace=True)
adrn=n.columns
adrn


# In[12]:


adr1_ninety = adr1[adrn]
adr2_ninety = adr2[adrn]
adr3_ninety = adr3[adrn]
adr4_ninety = adr4[adrn]


# In[13]:


adrTest1_ninety = adrTest1[adrn]
adrTest2_ninety = adrTest2[adrn]
adrTest3_ninety = adrTest3[adrn]
adrTest4_ninety = adrTest4[adrn]


# In[44]:


features = [feature1, feature2, feature3, feature4]
featureTests = [featureTest1, featureTest2, featureTest3, featureTest4]
adrs = [adr1_ninety, adr2_ninety, adr3_ninety, adr4_ninety]
adrTests = [adrTest1_ninety, adrTest2_ninety, adrTest3_ninety, adrTest4_ninety]
models = {"subset": [],
          "patience": [],
          "trainingAcc": [],
          "testingAcc": [],
          "trainAuc": [],
          "testAuc": [],
          "trainPre": [],
          "testPre": [],
          "trainRe": [],
          "testRe": [],
          "trainingLoss": [],
          "testingLoss": [],
          }


# In[46]:


counter = 0
for i in range(4):
    for p in [2,5,8,10,13,15,18,20]:
        model = Sequential()
        model.add(Dense(300, input_dim=11164, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(30, input_dim=11164, activation='tanh'))
        model.add(Dense(3, activation='sigmoid'))
        opt = SGD(lr=0.1, momentum=0.9)
        if counter==0:
            count = ""
        else:
            count = "_" + str(counter)
            
        es_callback = EarlyStopping(monitor='val_loss', patience=p)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(features[i], adrs[i] , validation_data=(featureTests[i], adrTests[i]), epochs=200, batch_size=10000, verbose=1, callbacks=[es_callback])
        models["trainingAcc"].append(np.mean(history.history['accuracy']))
        models["testingAcc"].append(np.mean(history.history['val_accuracy']))
        models["trainingLoss"].append(np.mean(history.history['loss']))
        models["testingLoss"].append(np.mean(history.history['val_loss']))
        models["trainAuc"].append(np.mean(history.history['auc'+str(counter)]))
        models["testAuc"].append(np.mean(history.history['val_auc'+str(counter)]))
        models["trainPre"].append(np.mean(history.history['precision'+str(counter)]))
        models["testPre"].append(np.mean(history.history['val_precision'+str(counter)]))
        models["trainRe"].append(np.mean(history.history['recall'+str(counter)]))
        models["testRe"].append(np.mean(history.history['val_recall'+str(counter)]))
        models["patience"].append(p)
        models["subset"].append(i)
        model_json = model.to_json()
        with open("subset_" + str(i+1) + "/model-ninety" + str(i) + "-" + str(p) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("subset_" + str(i+1) + "/model-ninety" + str(i) + "-" + str(p) + ".h5")
        print("Saved model to disk")
        counter += 1


# In[50]:


df = pd.DataFrame.from_dict(models)


# In[51]:


df.to_csv("ninety-df.csv")


# ### 70% Data

# In[67]:


s = pd.read_csv("ADR_combined_seventy.csv")
s.drop("Unnamed: 0", axis=1, inplace=True)
adrs=s.columns
adrs


# In[74]:


adr1_seventy = adr1[adrs]
adr2_seventy = adr2[adrs]
adr3_seventy = adr3[adrs]
adr4_seventy = adr4[adrs]


# In[75]:


adrTest1_seventy = adrTest1[adrs]
adrTest2_seventy = adrTest2[adrs]
adrTest3_seventy = adrTest3[adrs]
adrTest4_seventy = adrTest4[adrs]


# In[76]:


features = [feature1, feature2, feature3, feature4]
featureTests = [featureTest1, featureTest2, featureTest3, featureTest4]
adrz = [adr1_seventy, adr2_seventy, adr3_seventy, adr4_seventy]
adrTests = [adrTest1_seventy, adrTest2_seventy, adrTest3_seventy, adrTest4_seventy]
models = {"subset": [],
          "patience": [],
          "trainingAcc": [],
          "testingAcc": [],
          "trainAuc": [],
          "testAuc": [],
          "trainPre": [],
          "testPre": [],
          "trainRe": [],
          "testRe": [],
          "trainingLoss": [],
          "testingLoss": [],
          }


# In[78]:


counter = 0
for i in range(4):
    for p in [2,5,8,10,13,15,18,20]:
        model = Sequential()
        model.add(Dense(300, input_dim=11164, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(50, input_dim=11164, activation='tanh'))
        model.add(Dense(19, activation='sigmoid'))
        opt = SGD(lr=0.1, momentum=0.9)
        if counter==0:
            count = ""
        else:
            count = "_" + str(counter)
            
        es_callback = EarlyStopping(monitor='val_loss', patience=p)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        history = model.fit(features[i], adrs[i] , validation_data=(featureTests[i], adrTests[i]), epochs=200, batch_size=10000, verbose=1, callbacks=[es_callback])
        models["trainingAcc"].append(np.mean(history.history['accuracy']))
        models["testingAcc"].append(np.mean(history.history['val_accuracy']))
        models["trainingLoss"].append(np.mean(history.history['loss']))
        models["testingLoss"].append(np.mean(history.history['val_loss']))
        models["trainAuc"].append(np.mean(history.history['auc'+str(counter)]))
        models["testAuc"].append(np.mean(history.history['val_auc'+str(counter)]))
        models["trainPre"].append(np.mean(history.history['precision'+str(counter)]))
        models["testPre"].append(np.mean(history.history['val_precision'+str(counter)]))
        models["trainRe"].append(np.mean(history.history['recall'+str(counter)]))
        models["testRe"].append(np.mean(history.history['val_recall'+str(counter)]))
        models["patience"].append(p)
        models["subset"].append(i)
        model_json = model.to_json()
        with open("subset_" + str(i+1) + "/model-seventy" + str(i) + "-" + str(p) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("subset_" + str(i+1) + "/model-seventy" + str(i) + "-" + str(p) + ".h5")
        print("Saved model to disk")
        counter += 1


# In[79]:


dfs = pd.DataFrame.from_dict(models)


# In[80]:


dfs.to_csv("seventy-df.csv")

