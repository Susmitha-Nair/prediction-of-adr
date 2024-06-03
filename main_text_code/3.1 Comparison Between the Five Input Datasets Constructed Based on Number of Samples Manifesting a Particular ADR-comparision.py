#!/usr/bin/env python
# coding: utf-8

# # Section 3.1

# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


final10 = pd.read_csv("College/average-ten-f1-based.csv")
final30 = pd.read_csv("College/average-thirty-f1-based.csv")
final50 = pd.read_csv("College/average-fifty-f1-based.csv")
final70 = pd.read_csv("College/average-seventy-f1-based.csv")
final90 = pd.read_csv("College/average-ninety-f1-based.csv")

final10.drop("Unnamed: 0", axis=1, inplace=True)
final30.drop("Unnamed: 0", axis=1, inplace=True)
final50.drop("Unnamed: 0", axis=1, inplace=True)
final70.drop("Unnamed: 0", axis=1, inplace=True)
final90.drop("Unnamed: 0", axis=1, inplace=True)


# In[17]:


df = pd.DataFrame()
df = df.append(final10.mean(), ignore_index=True)
df = df.append(final30.mean(), ignore_index=True)
df = df.append(final50.mean(), ignore_index=True)
df = df.append(final70.mean(), ignore_index=True)
df = df.append(final90.mean(), ignore_index=True)
df["percent"] = [10,30,50,70,90]


# In[18]:


df


# In[19]:


ax = df.plot(x = 'percent', kind='line', figsize=(16,10), marker='.', markersize=10, title="Average scores for each model")
ax.set_ylabel("Scores")


# In[41]:


df = pd.DataFrame()
df = df.append([len(final10[final10['scoreF1']>0.65])], ignore_index=True)
df = df.append([len(final30[final30['scoreF1']>0.65])], ignore_index=True)
df = df.append([len(final50[final50['scoreF1']>0.65])], ignore_index=True)
df = df.append([len(final70[final70['scoreF1']>0.65])], ignore_index=True)
df = df.append([len(final90[final90['scoreF1']>0.65])], ignore_index=True)
df["percent"] = [10,30,50,70,90]


# In[42]:


df


# In[43]:


ax = df.plot(x = 'percent', kind='bar', figsize=(10,7), legend=False, colormap = "Paired", title="Number of ADR's with more than 0.65 F1 Score for each partition")
ax.set_ylabel("Scores")
ax.plot()


# In[ ]:




