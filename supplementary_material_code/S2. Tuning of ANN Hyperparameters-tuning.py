#!/usr/bin/env python
# coding: utf-8

# # Section 2.3.1

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Number of layers/neurons

# ## High
# #### ANN Performance

# In[14]:


high1 = pd.read_csv("sup/set-layers1.csv")
high2 = pd.read_csv("sup/set-layers2.csv")
high3 = pd.read_csv("sup/set-layers3.csv")
high4 = pd.read_csv("sup/set-layers4.csv")

high1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
high2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
high3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
high4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[19]:


annHighs = high1.append(high2, ignore_index=True)
annHighs = annHighs.append(high3, ignore_index=True)
annHighs = annHighs.append(high4, ignore_index=True)
annHighs = annHighs.mean()
annHighs


# #### Average scores of 4 datasets

# In[39]:


resultsHigh = pd.read_csv("average-high.csv")
resultsHigh.drop("Unnamed: 0", axis=1, inplace=True)
resultsHigh = resultsHigh.mean()
resultsHigh


# ## Medium
# #### ANN Performance

# In[111]:


annMeds = pd.read_csv("full-ann-df.csv")
annMeds.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
annMeds = annMeds.mean()
annMeds


# #### Average scores of 4 datasets

# In[37]:


resultsMed = pd.read_csv("average-ten-f1-based.csv")
resultsMed.drop("Unnamed: 0", axis=1, inplace=True)
resultsMed = resultsMed.mean()


# In[38]:


resultsMed


# ## Low
# #### ANN Performance

# In[42]:


low1 = pd.read_csv("sup/set-less-layers1.csv")
low2 = pd.read_csv("sup/set-less-layers2.csv")
low3 = pd.read_csv("sup/set-less-layers3.csv")
low4 = pd.read_csv("sup/set-less-layers4.csv")

low1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
low2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
low3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
low4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[43]:


annLows = low1.append(low2, ignore_index=True)
annLows = annLows.append(low3, ignore_index=True)
annLows = annLows.append(low4, ignore_index=True)
annLows = annLows.mean()
annLows


# #### Average scores of 4 datasets

# In[46]:


resultsLows = pd.read_csv("average-low.csv")
resultsLows.drop("Unnamed: 0", axis=1, inplace=True)
resultsLows = resultsLows.mean()


# In[47]:


resultsLows


# ## Comparing Models

# In[114]:


df = pd.DataFrame()
df['High'] = annHighs.values
df['Medium'] = annMeds.values
df['Low'] = annLows.values
df.index = annHighs.keys()


# In[116]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# ## Comparing Results

# In[71]:


df = pd.DataFrame(columns = ["High", "Medium", "Low"])
df['High'] = resultsHigh.values
df['Medium'] = resultsMed.values
df['Low'] = resultsLows.values


# In[73]:


df.index = ["Macro F1", "Accuracy", "Macro ROC"]


# In[93]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# # Activation Functions

# ### ANN
# #### Softmax

# In[95]:


softmax1 = pd.read_csv("sup/set-softmax-1.csv")
softmax2 = pd.read_csv("sup/set-softmax-2.csv")
softmax3 = pd.read_csv("sup/set-softmax-3.csv")
softmax4 = pd.read_csv("sup/set-softmax-4.csv")

softmax1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
softmax2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
softmax3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
softmax4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[96]:


softmax = softmax1.append(softmax2, ignore_index=True)
softmax = softmax.append(softmax3, ignore_index=True)
softmax = softmax.append(softmax4, ignore_index=True)
softmax = softmax.mean()
softmax


# ### Average scores of 4 datasets

# In[192]:


resultsSoftmax = pd.read_csv("average-softmax.csv")
resultsSoftmax.drop("Unnamed: 0", axis=1, inplace=True)
resultsSoftmax = resultsSoftmax.mean()
resultsSoftmax


# #### Elu

# In[99]:


elu1 = pd.read_csv("sup/set-elu-1.csv")
elu2 = pd.read_csv("sup/set-elu-2.csv")
elu3 = pd.read_csv("sup/set-elu-3.csv")
elu4 = pd.read_csv("sup/set-elu-4.csv")

elu1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
elu2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
elu3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
elu4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[100]:


elu = elu1.append(elu2, ignore_index=True)
elu = elu.append(elu3, ignore_index=True)
elu = elu.append(elu4, ignore_index=True)
elu = elu.mean()
elu


# ### Average scores of 4 datasets

# In[196]:


resultsElu = pd.read_csv("average-elu.csv")
resultsElu.drop("Unnamed: 0", axis=1, inplace=True)
resultsElu = resultsElu.mean()
resultsElu


# #### Sigmoid

# In[194]:


sigmoid1 = pd.read_csv("sup/set-sigmoid-1.csv")
sigmoid2 = pd.read_csv("sup/set-sigmoid-2.csv")
sigmoid3 = pd.read_csv("sup/set-sigmoid-3.csv")
sigmoid4 = pd.read_csv("sup/set-sigmoid-4.csv")

sigmoid1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
sigmoid2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
sigmoid3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
sigmoid4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[195]:


sigmoid = sigmoid1.append(sigmoid2, ignore_index=True)
sigmoid = sigmoid.append(sigmoid3, ignore_index=True)
sigmoid = sigmoid.append(sigmoid4, ignore_index=True)
sigmoid = sigmoid.mean()
sigmoid


# ### Average scores of 4 datasets

# In[197]:


resultsSigmoid = pd.read_csv("average-sigmoid.csv")
resultsSigmoid.drop("Unnamed: 0", axis=1, inplace=True)
resultsSigmoid = resultsSigmoid.mean()
resultsSigmoid


# #### Tanh

# In[103]:


tanh1 = pd.read_csv("sup/set-tanh-1.csv")
tanh2 = pd.read_csv("sup/set-tanh-2.csv")
tanh3 = pd.read_csv("sup/set-tanh-3.csv")
tanh4 = pd.read_csv("sup/set-tanh-4.csv")

tanh1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
tanh2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
tanh3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
tanh4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[104]:


tanh = tanh1.append(tanh2, ignore_index=True)
tanh = tanh.append(tanh3, ignore_index=True)
tanh = tanh.append(tanh4, ignore_index=True)
tanh = tanh.mean()
tanh


# ### Average scores of 4 datasets

# In[201]:


resultsTanh = pd.read_csv("average-tanh.csv")
resultsTanh.drop("Unnamed: 0", axis=1, inplace=True)
resultsTanh = resultsTanh.mean()
resultsTanh


# #### Relu

# In[ ]:


relu1 = pd.read_csv("sup/set-relu-1.csv")
relu2 = pd.read_csv("sup/set-relu-2.csv")
relu3 = pd.read_csv("sup/set-relu-3.csv")
relu4 = pd.read_csv("sup/set-relu-4.csv")

relu1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
relu2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
relu3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
relu4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[117]:


relu = tanh1.append(tanh2, ignore_index=True)
relu = tanh.append(tanh3, ignore_index=True)
relu = tanh.append(tanh4, ignore_index=True)
relu = tanh.mean()
relu


# ### Average scores for 4 datasets

# In[199]:


resultsRelu = pd.read_csv("average-ten-f1-based.csv")
resultsRelu.drop("Unnamed: 0", axis=1, inplace=True)
resultsRelu = resultsRelu.mean()
resultsRelu


# ## Comparing models

# In[121]:


df = pd.DataFrame()
df['Softmax'] = softmax.values
df['Sigmoid'] = sigmoid.values
df['Elu'] = elu.values
df['Relu'] = relu.values
df['Tanh'] = tanh.values
df.index = softmax.keys()


# In[122]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# ## Comparing Results

# In[203]:


df = pd.DataFrame()
df['Softmax'] = resultsSoftmax.values
df['Sigmoid'] = resultsSigmoid.values
df['Elu'] = resultsElu.values
df['Relu'] = resultsRelu.values
df['Tanh'] = resultsTanh.values
df.index = resultsSoftmax.keys()


# In[204]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# # Batch Size

# ## ANN Performance

# ### 100

# In[123]:


set100_1 = pd.read_csv("batch/set-batch-1001.csv")
set100_2 = pd.read_csv("batch/set-batch-1002.csv")
set100_3 = pd.read_csv("batch/set-batch-1003.csv")
set100_4 = pd.read_csv("batch/set-batch-1004.csv")

set100_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set100_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set100_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set100_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[124]:


set100 = set100_1.append(set100_2, ignore_index=True)
set100 = set100.append(set100_3, ignore_index=True)
set100 = set100.append(set100_4, ignore_index=True)
set100 = set100.mean()
set100


# #### Average scores of 4 datasets

# In[ ]:


results100 = pd.read_csv("average-100.csv")
results100.drop("Unnamed: 0", axis=1, inplace=True)
results100 = resultsHigh.mean()
results100


# ### 500

# In[125]:


set500_1 = pd.read_csv("batch/set-batch-5001.csv")
set500_2 = pd.read_csv("batch/set-batch-5002.csv")
set500_3 = pd.read_csv("batch/set-batch-5003.csv")
set500_4 = pd.read_csv("batch/set-batch-5004.csv")

set500_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set500_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set500_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set500_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[126]:


set500 = set500_1.append(set500_2, ignore_index=True)
set500 = set500.append(set500_3, ignore_index=True)
set500 = set500.append(set500_4, ignore_index=True)
set500 = set500.mean()
set500


# ### 1000

# In[127]:


set1000_1 = pd.read_csv("batch/set-batch-10001.csv")
set1000_2 = pd.read_csv("batch/set-batch-10002.csv")
set1000_3 = pd.read_csv("batch/set-batch-10003.csv")
set1000_4 = pd.read_csv("batch/set-batch-10004.csv")

set1000_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set1000_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set1000_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set1000_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[128]:


set1000 = set1000_1.append(set1000_2, ignore_index=True)
set1000 = set1000.append(set1000_3, ignore_index=True)
set1000 = set1000.append(set1000_4, ignore_index=True)
set1000 = set1000.mean()
set1000


# ### 5000

# In[129]:


set5000_1 = pd.read_csv("batch/set-batch-50001.csv")
set5000_2 = pd.read_csv("batch/set-batch-50002.csv")
set5000_3 = pd.read_csv("batch/set-batch-50003.csv")
set5000_4 = pd.read_csv("batch/set-batch-50004.csv")

set5000_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set5000_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set5000_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set5000_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[130]:


set5000 = set5000_1.append(set5000_2, ignore_index=True)
set5000 = set5000.append(set5000_3, ignore_index=True)
set5000 = set5000.append(set5000_4, ignore_index=True)
set5000 = set5000.mean()
set5000


# In[ ]:





# ### 10000

# In[131]:


set10000_1 = pd.read_csv("batch/set-batch-100001.csv")
set10000_2 = pd.read_csv("batch/set-batch-100002.csv")
set10000_3 = pd.read_csv("batch/set-batch-100003.csv")
set10000_4 = pd.read_csv("batch/set-batch-100004.csv")

set10000_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set10000_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set10000_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set10000_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[132]:


set10000 = set10000_1.append(set10000_2, ignore_index=True)
set10000 = set10000.append(set10000_3, ignore_index=True)
set10000 = set10000.append(set10000_4, ignore_index=True)
set10000 = set10000.mean()
set10000


# ### 20000

# In[133]:


set20000_1 = pd.read_csv("batch/set-batch-200001.csv")
set20000_2 = pd.read_csv("batch/set-batch-200002.csv")
set20000_3 = pd.read_csv("batch/set-batch-200003.csv")
set20000_4 = pd.read_csv("batch/set-batch-200004.csv")

set20000_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set20000_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set20000_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set20000_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[134]:


set20000 = set20000_1.append(set20000_2, ignore_index=True)
set20000 = set20000.append(set20000_3, ignore_index=True)
set20000 = set20000.append(set20000_4, ignore_index=True)
set20000 = set20000.mean()
set20000


# ### 30000

# In[135]:


set30000_1 = pd.read_csv("batch/set-batch-300001.csv")
set30000_2 = pd.read_csv("batch/set-batch-300002.csv")
set30000_3 = pd.read_csv("batch/set-batch-300003.csv")
set30000_4 = pd.read_csv("batch/set-batch-300004.csv")

set30000_1.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set30000_2.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set30000_3.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
set30000_4.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)


# In[136]:


set30000 = set30000_1.append(set30000_2, ignore_index=True)
set30000 = set30000.append(set30000_3, ignore_index=True)
set30000 = set30000.append(set30000_4, ignore_index=True)
set30000 = set30000.mean()
set30000


# In[162]:


df = pd.DataFrame()
df['100'] = set100.values
df['500'] = set500.values
df['1000'] = set1000.values
df['5000'] = set5000.values
df['10000'] = set10000.values
df['20000'] = set20000.values
df['30000'] = set30000.values
df.index = set100.keys()


# In[142]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# # Patience

# ### Model performance

# In[164]:


annMeds = pd.read_csv("full-ann-df.csv")
annMeds.drop(["Unnamed: 0", 'subset'], axis=1, inplace=True)
metricCols = annMeds.columns
df = annMeds.groupby('patience').mean()


# In[171]:


df = df.T


# In[200]:


ax = df.plot(kind='bar', colormap = "Set3_r", figsize=(16,10), title="Average scores for each model")
ax.set_ylabel("Scores")


# ### Average scores of 4 datasets

# In[ ]:




