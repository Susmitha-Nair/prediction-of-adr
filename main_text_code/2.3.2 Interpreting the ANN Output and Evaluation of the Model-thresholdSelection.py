#!/usr/bin/env python
# coding: utf-8

# # Section 2.3.2

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score


# ### Reading Data - 10%

# In[2]:


#Reading Actual Training Data
true_train1 = pd.read_csv("../../prediction_of_ADR/ADR_dataset_for_training_subset_1.csv")
true_train2 = pd.read_csv("../../prediction_of_ADR/ADR_dataset_for_training_subset_2.csv")
true_train3 = pd.read_csv("../../prediction_of_ADR/ADR_dataset_for_training_subset_3.csv")
true_train4 = pd.read_csv("../../prediction_of_ADR/ADR_dataset_for_training_subset_4.csv")

#Reading Predictions
pred_train1 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre1-ten-train.npy")
pred_train2 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre2-ten-train.npy")
pred_train3 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre3-ten-train.npy")
pred_train4 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre4-ten-train.npy")

true_train1.drop("Unnamed: 0", inplace=True, axis=1)
true_train2.drop("Unnamed: 0", inplace=True, axis=1)
true_train3.drop("Unnamed: 0", inplace=True, axis=1)
true_train4.drop("Unnamed: 0", inplace=True, axis=1)


# In[3]:


#Reading Actual Testing Data
true_test1 = pd.read_csv("../../prediction_of_ADR/ADR_validation_for_validation_subset_1.csv")
true_test2 = pd.read_csv("../../prediction_of_ADR/ADR_validation_for_validation_subset_2.csv")
true_test3 = pd.read_csv("../../prediction_of_ADR/ADR_validation_for_validation_subset_3.csv")
true_test4 = pd.read_csv("../../prediction_of_ADR/ADR_validation_for_validation_subset_4.csv")

#Reading Predictions
pred_test1 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre1-ten-test.npy")
pred_test2 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre2-ten-test.npy")
pred_test3 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre3-ten-test.npy")
pred_test4 = np.load("../../FinalData/3.1 Comparison Between the Five Input Datasets Constructed Based on Number of Samples Manifesting a Particular ADR/10/pre4-ten-test.npy")

true_test1.drop("Unnamed: 0", inplace=True, axis=1)
true_test2.drop("Unnamed: 0", inplace=True, axis=1)
true_test3.drop("Unnamed: 0", inplace=True, axis=1)
true_test4.drop("Unnamed: 0", inplace=True, axis=1)


# In[4]:


true_train_labels1 = true_train1.values
true_train_labels2 = true_train2.values
true_train_labels3 = true_train3.values
true_train_labels4 = true_train4.values

true_train_labels1 = true_train_labels1.T
true_train_labels2 = true_train_labels2.T
true_train_labels3 = true_train_labels3.T
true_train_labels4 = true_train_labels4.T

print(true_train_labels1.shape)
print(true_train_labels2.shape)
print(true_train_labels3.shape)
print(true_train_labels4.shape)


true_test_labels1 = true_test1.values
true_test_labels2 = true_test2.values
true_test_labels3 = true_test3.values
true_test_labels4 = true_test4.values

true_test_labels1 = true_test_labels1.T
true_test_labels2 = true_test_labels2.T
true_test_labels3 = true_test_labels3.T
true_test_labels4 = true_test_labels4.T

print(true_test_labels1.shape)
print(true_test_labels2.shape)
print(true_test_labels3.shape)
print(true_test_labels4.shape)


# In[5]:


pred_train1 = pred_train1.T
pred_train2 = pred_train2.T
pred_train3 = pred_train3.T
pred_train4 = pred_train4.T

print(pred_train1.shape)
print(pred_train2.shape)
print(pred_train3.shape)
print(pred_train4.shape)


pred_test1 = pred_test1.T
pred_test2 = pred_test2.T
pred_test3 = pred_test3.T
pred_test4 = pred_test4.T

print(pred_test1.shape)
print(pred_test2.shape)
print(pred_test3.shape)
print(pred_test4.shape)


# ### Getting Thresholds

# In[8]:


adrs = list(true_train1.columns)
trueLabelsTrain = [true_train_labels1, true_train_labels2, true_train_labels3, true_train_labels4]
trueLabelsTest = [true_test_labels1, true_test_labels2, true_test_labels3, true_test_labels4]
predTrain = [pred_train1, pred_train2, pred_train3, pred_train4]
predTest = [pred_test1, pred_test2, pred_test3, pred_test4]


# In[34]:


def getAllthresholds(pred):
    thresholds = {}
    index=0
    for adr in adrs:
        thresholds[adr] = np.linspace(min(pred[index]),max(pred[index]),100)
        index+=1
    return thresholds


# ### Choosing best thresholds

# In[35]:


def getBestThresholds(true_labels, pred, thresholds):
    selectedThreshF1 = []

    for adr in range(243):
        maxThreshF1 = 0
        maxScoreF1 = 0

        for threshold in thresholds[adrs[adr]]:
            temp = []
            #Using training data only
            for sample in range(len(pred[adr])):
                if pred[adr][sample] > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            scoreF1 = f1_score(true_labels[adr], temp, average= "macro")

            if scoreF1 > maxScoreF1:
                maxThreshF1 = threshold
                maxScoreF1 = scoreF1

        selectedThreshF1.append(maxThreshF1)   
    return selectedThreshF1


# In[37]:


selectedThresholds = []
for i in range(4):
    thresholds = getAllthresholds(predTrain[i])
    selectedThresholds.append(getBestThresholds(trueLabelsTrain[i], predTrain[i], thresholds))


# In[38]:


for i in range(4):
    print(len(selectedThresholds[i]))


# In[45]:


np.save("../../FinalData/selectedThresh.npy", selectedThresholds)


# In[6]:


selectedThresholds = np.load("../../FinalData/selectedThresh.npy")


# ### Applying thresholds to testing data

# In[9]:


scores = []
for i in range(4):
    score = {'scoreF1': [], 'scoreAcc': [], 'scoreRoc': []}
    pred_test = predTest[i]
    thresholds = selectedThresholds[i]
    truelabels = trueLabelsTest[i]
    for adr in range(243):
        temp = []
        for sample in range(2000):
            if pred_test[adr][sample] > thresholds[adr]:
                temp.append(1)
            else:
                temp.append(0)
        
        score['scoreF1'].append(f1_score(truelabels[adr], temp, average= "macro"))
        score['scoreAcc'].append(accuracy_score(truelabels[adr], temp))
        score['scoreRoc'].append(roc_auc_score(truelabels[adr], temp, average= "macro"))
    scores.append(score)


# In[10]:


len(thresholds)


# In[11]:


df1 = pd.DataFrame.from_dict(scores[0])
df2 = pd.DataFrame.from_dict(scores[1])
df3 = pd.DataFrame.from_dict(scores[2])
df4 = pd.DataFrame.from_dict(scores[3])


# In[12]:


df = pd.DataFrame(columns = df1.columns)
df["ADR"] = adrs
for i in range(243):
    for j in range(3):
        df.iloc[i,j] = (df1.iloc[i,j]+df2.iloc[i,j]+df3.iloc[i,j]+df4.iloc[i,j])/4


# In[43]:


df.sort_values(by=['scoreF1'], ascending=False, inplace=True)
df


# In[20]:


df[:9].mean()


# In[41]:


df.to_csv("average-ten-f1-based.csv")


# ### Reading Data - 70%

# In[59]:


seventy = pd.read_csv("ADR_combined_seventy.csv")
seventy.drop("Unnamed: 0", axis=1, inplace=True)


# In[60]:


#Reading Actual Training Data
true_train1 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_1.csv")
true_train2 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_2.csv")
true_train3 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_3.csv")
true_train4 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_4.csv")

#Reading Predictions
pred_train1 = np.load("pre1-s-train.npy")
pred_train2 = np.load("pre2-s-train.npy")
pred_train3 = np.load("pre3-s-train.npy")
pred_train4 = np.load("pre4-s-train.npy")

true_train1.drop("Unnamed: 0", inplace=True, axis=1)
true_train2.drop("Unnamed: 0", inplace=True, axis=1)
true_train3.drop("Unnamed: 0", inplace=True, axis=1)
true_train4.drop("Unnamed: 0", inplace=True, axis=1)


# In[61]:


#Reading Actual Testing Data
true_test1 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_1.csv")
true_test2 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_2.csv")
true_test3 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_3.csv")
true_test4 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_4.csv")

#Reading Predictions
pred_test1 = np.load("pre1-s.npy")
pred_test2 = np.load("pre2-s.npy")
pred_test3 = np.load("pre3-s.npy")
pred_test4 = np.load("pre4-s.npy")

true_test1.drop("Unnamed: 0", inplace=True, axis=1)
true_test2.drop("Unnamed: 0", inplace=True, axis=1)
true_test3.drop("Unnamed: 0", inplace=True, axis=1)
true_test4.drop("Unnamed: 0", inplace=True, axis=1)


# In[62]:


true_train1 = true_train1[seventy.columns]
true_train2 = true_train2[seventy.columns]
true_train3 = true_train3[seventy.columns]
true_train4 = true_train4[seventy.columns]

true_test1 = true_test1[seventy.columns]
true_test2 = true_test2[seventy.columns]
true_test3 = true_test3[seventy.columns]
true_test4 = true_test4[seventy.columns]


# In[63]:


true_train_labels1 = true_train1.values
true_train_labels2 = true_train2.values
true_train_labels3 = true_train3.values
true_train_labels4 = true_train4.values

true_train_labels1 = true_train_labels1.T
true_train_labels2 = true_train_labels2.T
true_train_labels3 = true_train_labels3.T
true_train_labels4 = true_train_labels4.T

print(true_train_labels1.shape)
print(true_train_labels2.shape)
print(true_train_labels3.shape)
print(true_train_labels4.shape)


true_test_labels1 = true_test1.values
true_test_labels2 = true_test2.values
true_test_labels3 = true_test3.values
true_test_labels4 = true_test4.values

true_test_labels1 = true_test_labels1.T
true_test_labels2 = true_test_labels2.T
true_test_labels3 = true_test_labels3.T
true_test_labels4 = true_test_labels4.T

print(true_test_labels1.shape)
print(true_test_labels2.shape)
print(true_test_labels3.shape)
print(true_test_labels4.shape)


# In[64]:


pred_train1 = pred_train1.T
pred_train2 = pred_train2.T
pred_train3 = pred_train3.T
pred_train4 = pred_train4.T

print(pred_train1.shape)
print(pred_train2.shape)
print(pred_train3.shape)
print(pred_train4.shape)


pred_test1 = pred_test1.T
pred_test2 = pred_test2.T
pred_test3 = pred_test3.T
pred_test4 = pred_test4.T

print(pred_test1.shape)
print(pred_test2.shape)
print(pred_test3.shape)
print(pred_test4.shape)


# ### Getting Thresholds

# In[ ]:


adrs = list(true_train1.columns)
trueLabelsTrain = [true_train_labels1, true_train_labels2, true_train_labels3, true_train_labels4]
trueLabelsTest = [true_test_labels1, true_test_labels2, true_test_labels3, true_test_labels4]
predTrain = [pred_train1, pred_train2, pred_train3, pred_train4]
predTest = [pred_test1, pred_test2, pred_test3, pred_test4]


# In[66]:


def getAllthresholds(pred):
    thresholds = {}
    index=0
    for adr in adrs:
        thresholds[adr] = np.linspace(min(pred[index]),max(pred[index]),100)
        index+=1
    return thresholds


# ### Choosing best thresholds

# In[67]:


def getBestThresholds(true_labels, pred, thresholds):
    selectedThreshF1 = []

    for adr in range(19):
        maxThreshF1 = 0
        maxScoreF1 = 0

        for threshold in thresholds[adrs[adr]]:
            temp = []
            #Using training data only
            for sample in range(len(pred[adr])):
                if pred[adr][sample] > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            scoreF1 = f1_score(true_labels[adr], temp, average= "macro")

            if scoreF1 > maxScoreF1:
                maxThreshF1 = threshold
                maxScoreF1 = scoreF1

        selectedThreshF1.append(maxThreshF1)   
    return selectedThreshF1


# In[68]:


selectedThresholds = []
for i in range(4):
    thresholds = getAllthresholds(predTrain[i])
    selectedThresholds.append(getBestThresholds(trueLabelsTrain[i], predTrain[i], thresholds))


# In[69]:


for i in range(4):
    print(len(selectedThresholds[i]))


# ### Adding thresholds to test data

# In[70]:


scores = []
for i in range(4):
    score = {'scoreF1': [], 'scoreAcc': [], 'scoreRoc': []}
    pred_test = predTest[i]
    thresholds = selectedThresholds[i]
    truelabels = trueLabelsTest[i]
    for adr in range(19):
        temp = []
        for sample in range(2000):
            if pred_test[adr][sample] > thresholds[adr]:
                temp.append(1)
            else:
                temp.append(0)
        
        score['scoreF1'].append(f1_score(truelabels[adr], temp, average= "macro"))
        score['scoreAcc'].append(accuracy_score(truelabels[adr], temp))
        score['scoreRoc'].append(roc_auc_score(truelabels[adr], temp, average= "macro"))
    scores.append(score)


# In[71]:


df1 = pd.DataFrame.from_dict(scores[0])
df2 = pd.DataFrame.from_dict(scores[1])
df3 = pd.DataFrame.from_dict(scores[2])
df4 = pd.DataFrame.from_dict(scores[3])


# In[73]:


df = pd.DataFrame(columns = df1.columns)
df["ADR"] = adrs
for i in range(19):
    for j in range(3):
        df.iloc[i,j] = (df1.iloc[i,j]+df2.iloc[i,j]+df3.iloc[i,j]+df4.iloc[i,j])/4


# In[74]:


df.sort_values(by=['scoreF1'], ascending=False, inplace=True)
df


# In[75]:


df.to_csv("average-seventy-f1-based.csv")


# ### Reading Data - 90%

# In[81]:


ninety = pd.read_csv("ADR_combined_ninety.csv")
ninety.drop("Unnamed: 0", axis=1, inplace=True)


# In[77]:


#Reading Actual Training Data
true_train1 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_1.csv")
true_train2 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_2.csv")
true_train3 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_3.csv")
true_train4 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_4.csv")

#Reading Predictions
pred_train1 = np.load("pre1-n-train.npy")
pred_train2 = np.load("pre2-n-train.npy")
pred_train3 = np.load("pre3-n-train.npy")
pred_train4 = np.load("pre4-n-train.npy")

true_train1.drop("Unnamed: 0", inplace=True, axis=1)
true_train2.drop("Unnamed: 0", inplace=True, axis=1)
true_train3.drop("Unnamed: 0", inplace=True, axis=1)
true_train4.drop("Unnamed: 0", inplace=True, axis=1)


# In[79]:


#Reading Actual Testing Data
true_test1 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_1.csv")
true_test2 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_2.csv")
true_test3 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_3.csv")
true_test4 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_4.csv")

#Reading Predictions
pred_test1 = np.load("pre1-n.npy")
pred_test2 = np.load("pre2-n.npy")
pred_test3 = np.load("pre3-n.npy")
pred_test4 = np.load("pre4-n.npy")

true_test1.drop("Unnamed: 0", inplace=True, axis=1)
true_test2.drop("Unnamed: 0", inplace=True, axis=1)
true_test3.drop("Unnamed: 0", inplace=True, axis=1)
true_test4.drop("Unnamed: 0", inplace=True, axis=1)


# In[82]:


true_train1 = true_train1[ninety.columns]
true_train2 = true_train2[ninety.columns]
true_train3 = true_train3[ninety.columns]
true_train4 = true_train4[ninety.columns]

true_test1 = true_test1[ninety.columns]
true_test2 = true_test2[ninety.columns]
true_test3 = true_test3[ninety.columns]
true_test4 = true_test4[ninety.columns]


# In[83]:


true_train_labels1 = true_train1.values
true_train_labels2 = true_train2.values
true_train_labels3 = true_train3.values
true_train_labels4 = true_train4.values

true_train_labels1 = true_train_labels1.T
true_train_labels2 = true_train_labels2.T
true_train_labels3 = true_train_labels3.T
true_train_labels4 = true_train_labels4.T

print(true_train_labels1.shape)
print(true_train_labels2.shape)
print(true_train_labels3.shape)
print(true_train_labels4.shape)


true_test_labels1 = true_test1.values
true_test_labels2 = true_test2.values
true_test_labels3 = true_test3.values
true_test_labels4 = true_test4.values

true_test_labels1 = true_test_labels1.T
true_test_labels2 = true_test_labels2.T
true_test_labels3 = true_test_labels3.T
true_test_labels4 = true_test_labels4.T

print(true_test_labels1.shape)
print(true_test_labels2.shape)
print(true_test_labels3.shape)
print(true_test_labels4.shape)


# In[84]:


pred_train1 = pred_train1.T
pred_train2 = pred_train2.T
pred_train3 = pred_train3.T
pred_train4 = pred_train4.T

print(pred_train1.shape)
print(pred_train2.shape)
print(pred_train3.shape)
print(pred_train4.shape)


pred_test1 = pred_test1.T
pred_test2 = pred_test2.T
pred_test3 = pred_test3.T
pred_test4 = pred_test4.T

print(pred_test1.shape)
print(pred_test2.shape)
print(pred_test3.shape)
print(pred_test4.shape)


# ### Getting thresholds

# In[85]:


adrs = list(true_train1.columns)
trueLabelsTrain = [true_train_labels1, true_train_labels2, true_train_labels3, true_train_labels4]
trueLabelsTest = [true_test_labels1, true_test_labels2, true_test_labels3, true_test_labels4]
predTrain = [pred_train1, pred_train2, pred_train3, pred_train4]
predTest = [pred_test1, pred_test2, pred_test3, pred_test4]


# In[86]:


def getAllthresholds(pred):
    thresholds = {}
    index=0
    for adr in adrs:
        thresholds[adr] = np.linspace(min(pred[index]),max(pred[index]),100)
        index+=1
    return thresholds


# ### Choosing best thresholds

# In[87]:


def getBestThresholds(true_labels, pred, thresholds):
    selectedThreshF1 = []

    for adr in range(3):
        maxThreshF1 = 0
        maxScoreF1 = 0

        for threshold in thresholds[adrs[adr]]:
            temp = []
            #Using training data only
            for sample in range(len(pred[adr])):
                if pred[adr][sample] > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            scoreF1 = f1_score(true_labels[adr], temp, average= "macro")

            if scoreF1 > maxScoreF1:
                maxThreshF1 = threshold
                maxScoreF1 = scoreF1

        selectedThreshF1.append(maxThreshF1)   
    return selectedThreshF1


# In[88]:


selectedThresholds = []
for i in range(4):
    thresholds = getAllthresholds(predTrain[i])
    selectedThresholds.append(getBestThresholds(trueLabelsTrain[i], predTrain[i], thresholds))


# In[89]:


for i in range(4):
    print(len(selectedThresholds[i]))


# ### Adding thresholds to test data

# In[91]:


scores = []
for i in range(4):
    score = {'scoreF1': [], 'scoreAcc': [], 'scoreRoc': []}
    pred_test = predTest[i]
    thresholds = selectedThresholds[i]
    truelabels = trueLabelsTest[i]
    for adr in range(3):
        temp = []
        for sample in range(2000):
            if pred_test[adr][sample] > thresholds[adr]:
                temp.append(1)
            else:
                temp.append(0)
        
        score['scoreF1'].append(f1_score(truelabels[adr], temp, average= "macro"))
        score['scoreAcc'].append(accuracy_score(truelabels[adr], temp))
        score['scoreRoc'].append(roc_auc_score(truelabels[adr], temp, average= "macro"))
    scores.append(score)


# In[92]:


df1 = pd.DataFrame.from_dict(scores[0])
df2 = pd.DataFrame.from_dict(scores[1])
df3 = pd.DataFrame.from_dict(scores[2])
df4 = pd.DataFrame.from_dict(scores[3])


# In[93]:


df = pd.DataFrame(columns = df1.columns)
df["ADR"] = adrs
for i in range(3):
    for j in range(3):
        df.iloc[i,j] = (df1.iloc[i,j]+df2.iloc[i,j]+df3.iloc[i,j]+df4.iloc[i,j])/4


# In[94]:


df.sort_values(by=['scoreF1'], ascending=False, inplace=True)
df


# In[95]:


df.to_csv("average-ninety-f1-based.csv")


# In[5]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score


# ### Reading Data - 30%

# In[14]:


thirty = pd.read_excel("thirty_percent.xlsx")


# In[6]:


#Reading Actual Training Data
true_train1 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_1.csv")
true_train2 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_2.csv")
true_train3 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_3.csv")
true_train4 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_4.csv")

#Reading Predictions
pred_train1 = np.load("pre1-t-train.npy")
pred_train2 = np.load("pre2-t-train.npy")
pred_train3 = np.load("pre3-t-train.npy")
pred_train4 = np.load("pre4-t-train.npy")

true_train1.drop("Unnamed: 0", inplace=True, axis=1)
true_train2.drop("Unnamed: 0", inplace=True, axis=1)
true_train3.drop("Unnamed: 0", inplace=True, axis=1)
true_train4.drop("Unnamed: 0", inplace=True, axis=1)


# In[7]:


#Reading Actual Testing Data
true_test1 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_1.csv")
true_test2 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_2.csv")
true_test3 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_3.csv")
true_test4 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_4.csv")

#Reading Predictions
pred_test1 = np.load("pre1-t.npy")
pred_test2 = np.load("pre2-t.npy")
pred_test3 = np.load("pre3-t.npy")
pred_test4 = np.load("pre4-t.npy")

true_test1.drop("Unnamed: 0", inplace=True, axis=1)
true_test2.drop("Unnamed: 0", inplace=True, axis=1)
true_test3.drop("Unnamed: 0", inplace=True, axis=1)
true_test4.drop("Unnamed: 0", inplace=True, axis=1)


# In[15]:


true_train1 = true_train1[thirty['ADR']]
true_train2 = true_train2[thirty['ADR']]
true_train3 = true_train3[thirty['ADR']]
true_train4 = true_train4[thirty['ADR']]

true_test1 = true_test1[thirty['ADR']]
true_test2 = true_test2[thirty['ADR']]
true_test3 = true_test3[thirty['ADR']]
true_test4 = true_test4[thirty['ADR']]


# In[16]:


true_train_labels1 = true_train1.values
true_train_labels2 = true_train2.values
true_train_labels3 = true_train3.values
true_train_labels4 = true_train4.values

true_train_labels1 = true_train_labels1.T
true_train_labels2 = true_train_labels2.T
true_train_labels3 = true_train_labels3.T
true_train_labels4 = true_train_labels4.T

print(true_train_labels1.shape)
print(true_train_labels2.shape)
print(true_train_labels3.shape)
print(true_train_labels4.shape)


true_test_labels1 = true_test1.values
true_test_labels2 = true_test2.values
true_test_labels3 = true_test3.values
true_test_labels4 = true_test4.values

true_test_labels1 = true_test_labels1.T
true_test_labels2 = true_test_labels2.T
true_test_labels3 = true_test_labels3.T
true_test_labels4 = true_test_labels4.T

print(true_test_labels1.shape)
print(true_test_labels2.shape)
print(true_test_labels3.shape)
print(true_test_labels4.shape)


# In[18]:


pred_train1 = pred_train1.T
pred_train2 = pred_train2.T
pred_train3 = pred_train3.T
pred_train4 = pred_train4.T

print(pred_train1.shape)
print(pred_train2.shape)
print(pred_train3.shape)
print(pred_train4.shape)


pred_test1 = pred_test1.T
pred_test2 = pred_test2.T
pred_test3 = pred_test3.T
pred_test4 = pred_test4.T

print(pred_test1.shape)
print(pred_test2.shape)
print(pred_test3.shape)
print(pred_test4.shape)


# ### Getting Thresholds

# In[19]:


adrs = list(true_train1.columns)
trueLabelsTrain = [true_train_labels1, true_train_labels2, true_train_labels3, true_train_labels4]
trueLabelsTest = [true_test_labels1, true_test_labels2, true_test_labels3, true_test_labels4]
predTrain = [pred_train1, pred_train2, pred_train3, pred_train4]
predTest = [pred_test1, pred_test2, pred_test3, pred_test4]


# In[20]:


def getAllthresholds(pred):
    thresholds = {}
    index=0
    for adr in adrs:
        thresholds[adr] = np.linspace(min(pred[index]),max(pred[index]),100)
        index+=1
    return thresholds


# ### Choosing best thresholds

# In[25]:


def getBestThresholds(true_labels, pred, thresholds):
    selectedThreshF1 = []

    for adr in range(165):
        maxThreshF1 = 0
        maxScoreF1 = 0

        for threshold in thresholds[adrs[adr]]:
            temp = []
            #Using training data only
            for sample in range(len(pred[adr])):
                if pred[adr][sample] > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            scoreF1 = f1_score(true_labels[adr], temp, average= "macro")

            if scoreF1 > maxScoreF1:
                maxThreshF1 = threshold
                maxScoreF1 = scoreF1

        selectedThreshF1.append(maxThreshF1)   
    return selectedThreshF1


# In[26]:


selectedThresholds = []
for i in range(4):
    thresholds = getAllthresholds(predTrain[i])
    selectedThresholds.append(getBestThresholds(trueLabelsTrain[i], predTrain[i], thresholds))


# In[29]:


for i in range(4):
    print(len(selectedThresholds[i]))


# ### Applying thresholds to test data

# In[30]:


scores = []
for i in range(4):
    score = {'scoreF1': [], 'scoreAcc': [], 'scoreRoc': []}
    pred_test = predTest[i]
    thresholds = selectedThresholds[i]
    truelabels = trueLabelsTest[i]
    for adr in range(165):
        temp = []
        for sample in range(2000):
            if pred_test[adr][sample] > thresholds[adr]:
                temp.append(1)
            else:
                temp.append(0)
        
        score['scoreF1'].append(f1_score(truelabels[adr], temp, average= "macro"))
        score['scoreAcc'].append(accuracy_score(truelabels[adr], temp))
        score['scoreRoc'].append(roc_auc_score(truelabels[adr], temp, average= "macro"))
    scores.append(score)


# In[31]:


df1 = pd.DataFrame.from_dict(scores[0])
df2 = pd.DataFrame.from_dict(scores[1])
df3 = pd.DataFrame.from_dict(scores[2])
df4 = pd.DataFrame.from_dict(scores[3])


# In[41]:


df = pd.DataFrame(columns = df1.columns)
df["ADR"] = adrs
for i in range(165):
    for j in range(3):
        df.iloc[i,j] = (df1.iloc[i,j]+df2.iloc[i,j]+df3.iloc[i,j]+df4.iloc[i,j])/4


# In[44]:


df.sort_values(by=['scoreF1'], ascending=False, inplace=True)
df


# In[46]:


df.to_csv("average-thirty-f1-based.csv")


# ### Reading Data - 50%

# In[47]:


thirty = pd.read_excel("fifty_percent.xlsx")


# In[48]:


#Reading Actual Training Data
true_train1 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_1.csv")
true_train2 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_2.csv")
true_train3 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_3.csv")
true_train4 = pd.read_csv("prediction_of_ADR/ADR_dataset_for_training_subset_4.csv")

#Reading Predictions
pred_train1 = np.load("pre1-f-train.npy")
pred_train2 = np.load("pre2-f-train.npy")
pred_train3 = np.load("pre3-f-train.npy")
pred_train4 = np.load("pre4-f-train.npy")

true_train1.drop("Unnamed: 0", inplace=True, axis=1)
true_train2.drop("Unnamed: 0", inplace=True, axis=1)
true_train3.drop("Unnamed: 0", inplace=True, axis=1)
true_train4.drop("Unnamed: 0", inplace=True, axis=1)


# In[49]:


#Reading Actual Testing Data
true_test1 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_1.csv")
true_test2 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_2.csv")
true_test3 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_3.csv")
true_test4 = pd.read_csv("prediction_of_ADR/ADR_validation_for_validation_subset_4.csv")

#Reading Predictions
pred_test1 = np.load("pre1-f.npy")
pred_test2 = np.load("pre2-f.npy")
pred_test3 = np.load("pre3-f.npy")
pred_test4 = np.load("pre4-f.npy")

true_test1.drop("Unnamed: 0", inplace=True, axis=1)
true_test2.drop("Unnamed: 0", inplace=True, axis=1)
true_test3.drop("Unnamed: 0", inplace=True, axis=1)
true_test4.drop("Unnamed: 0", inplace=True, axis=1)


# In[50]:


true_train1 = true_train1[thirty['ADR']]
true_train2 = true_train2[thirty['ADR']]
true_train3 = true_train3[thirty['ADR']]
true_train4 = true_train4[thirty['ADR']]

true_test1 = true_test1[thirty['ADR']]
true_test2 = true_test2[thirty['ADR']]
true_test3 = true_test3[thirty['ADR']]
true_test4 = true_test4[thirty['ADR']]


# In[51]:


true_train_labels1 = true_train1.values
true_train_labels2 = true_train2.values
true_train_labels3 = true_train3.values
true_train_labels4 = true_train4.values

true_train_labels1 = true_train_labels1.T
true_train_labels2 = true_train_labels2.T
true_train_labels3 = true_train_labels3.T
true_train_labels4 = true_train_labels4.T

print(true_train_labels1.shape)
print(true_train_labels2.shape)
print(true_train_labels3.shape)
print(true_train_labels4.shape)


true_test_labels1 = true_test1.values
true_test_labels2 = true_test2.values
true_test_labels3 = true_test3.values
true_test_labels4 = true_test4.values

true_test_labels1 = true_test_labels1.T
true_test_labels2 = true_test_labels2.T
true_test_labels3 = true_test_labels3.T
true_test_labels4 = true_test_labels4.T

print(true_test_labels1.shape)
print(true_test_labels2.shape)
print(true_test_labels3.shape)
print(true_test_labels4.shape)


# In[52]:


pred_train1 = pred_train1.T
pred_train2 = pred_train2.T
pred_train3 = pred_train3.T
pred_train4 = pred_train4.T

print(pred_train1.shape)
print(pred_train2.shape)
print(pred_train3.shape)
print(pred_train4.shape)


pred_test1 = pred_test1.T
pred_test2 = pred_test2.T
pred_test3 = pred_test3.T
pred_test4 = pred_test4.T

print(pred_test1.shape)
print(pred_test2.shape)
print(pred_test3.shape)
print(pred_test4.shape)


# ### Getting Thresholds

# In[53]:


adrs = list(true_train1.columns)
trueLabelsTrain = [true_train_labels1, true_train_labels2, true_train_labels3, true_train_labels4]
trueLabelsTest = [true_test_labels1, true_test_labels2, true_test_labels3, true_test_labels4]
predTrain = [pred_train1, pred_train2, pred_train3, pred_train4]
predTest = [pred_test1, pred_test2, pred_test3, pred_test4]


# In[54]:


def getAllthresholds(pred):
    thresholds = {}
    index=0
    for adr in adrs:
        thresholds[adr] = np.linspace(min(pred[index]),max(pred[index]),100)
        index+=1
    return thresholds


# ### Choosing best thresholds

# In[56]:


def getBestThresholds(true_labels, pred, thresholds):
    selectedThreshF1 = []

    for adr in range(68):
        maxThreshF1 = 0
        maxScoreF1 = 0

        for threshold in thresholds[adrs[adr]]:
            temp = []
            #Using training data only
            for sample in range(len(pred[adr])):
                if pred[adr][sample] > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            scoreF1 = f1_score(true_labels[adr], temp, average= "macro")

            if scoreF1 > maxScoreF1:
                maxThreshF1 = threshold
                maxScoreF1 = scoreF1

        selectedThreshF1.append(maxThreshF1)   
    return selectedThreshF1


# In[57]:


selectedThresholds = []
for i in range(4):
    thresholds = getAllthresholds(predTrain[i])
    selectedThresholds.append(getBestThresholds(trueLabelsTrain[i], predTrain[i], thresholds))


# In[58]:


for i in range(4):
    print(len(selectedThresholds[i]))


# ### Adding thresholds to test data

# In[59]:


scores = []
for i in range(4):
    score = {'scoreF1': [], 'scoreAcc': [], 'scoreRoc': []}
    pred_test = predTest[i]
    thresholds = selectedThresholds[i]
    truelabels = trueLabelsTest[i]
    for adr in range(68):
        temp = []
        for sample in range(2000):
            if pred_test[adr][sample] > thresholds[adr]:
                temp.append(1)
            else:
                temp.append(0)
        
        score['scoreF1'].append(f1_score(truelabels[adr], temp, average= "macro"))
        score['scoreAcc'].append(accuracy_score(truelabels[adr], temp))
        score['scoreRoc'].append(roc_auc_score(truelabels[adr], temp, average= "macro"))
    scores.append(score)


# In[60]:


df1 = pd.DataFrame.from_dict(scores[0])
df2 = pd.DataFrame.from_dict(scores[1])
df3 = pd.DataFrame.from_dict(scores[2])
df4 = pd.DataFrame.from_dict(scores[3])


# In[61]:


df = pd.DataFrame(columns = df1.columns)
df["ADR"] = adrs
for i in range(68):
    for j in range(3):
        df.iloc[i,j] = (df1.iloc[i,j]+df2.iloc[i,j]+df3.iloc[i,j]+df4.iloc[i,j])/4


# In[62]:


df.sort_values(by=['scoreF1'], ascending=False, inplace=True)
df


# In[63]:


df.to_csv("average-fifty-f1-based.csv")


# In[22]:


df = pd.read_csv("../../../Datasets/average-ten-f1-based.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)


# In[23]:


df_avg = df.mean()
df_avg


# In[ ]:




