#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import re
import pickle

def imageCleanup(imageUrls):
    imageUrls = imageUrls.split('\'')
    for i in imageUrls:
        if len(i) > 20:
            if 'png' in i:
                return i
def idCleanup(id):
    id = id.replace(" ","")
    return id[id.find("[")+1:id.find("]")]

def nameCleaner(input):
    input = input.replace(" ","_")
    return re.sub("[^A-Z+_]", "", input,0,re.IGNORECASE)

def create_training_data():
    for img in os.listdir(dataDir):
        try:
            img_array = cv2.imread(os.path.join(dataDir,img))
            training_data.append([img_array,categories.index(img.split("-")[0])])
        except Exception as e:
            pass


# In[2]:


cols = pd.read_csv('cards.csv', nrows=1).columns
df = pd.read_csv('cards.csv', usecols=cols, low_memory=False)
desiredColumns = ['power', 'toughness', 'type_line', 'keywords', 'name','oracle_text','image_uris', 'multiverse_ids']
df.drop(df.columns.difference(desiredColumns), axis=1, inplace=True)
df = df[df['image_uris'].notna()]
df = df[df['name'].notna()]
df['image_uris'] = df.image_uris.apply(imageCleanup)
df['multiverse_ids'] = df.multiverse_ids.apply(idCleanup)
df['name'] = df.name.apply(nameCleaner)


# In[3]:


uniqueDF = df
uniqueDF.drop_duplicates(subset="name",keep="first", inplace=True)
categories  = list(uniqueDF["name"])
categoriesID = list(uniqueDF["multiverse_ids"])


# In[4]:


training_data = []
dataDir = "E:/Jupyter"
create_training_data()


# In[6]:


X = []
Y = []


# In[7]:


for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, 200, 280, 3)


# In[13]:


pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()

pickle_out = open("Cat.pickle","wb")
pickle.dump(categories,pickle_out)
pickle_out.close()


# In[57]:


pickle_out = open("X2.pickle","wb")
pickle.dump(X[:int(len(X)/2)],pickle_out)
pickle_out.close()

reducedY = Y[:int(len(Y)/2)]
reducedCat = []
for index in range(len(reducedY)):
    target = reducedY[index]
    if categories[target] in reducedCat:
        if reducedCat.index(categories[target]) == 12189:
            print("holla")
        reducedY[index] = reducedCat.index(categories[target])
    else:
        reducedCat.append(categories[target])
        reducedY[index] = len(reducedCat)-1

pickle_out = open("Y2.pickle","wb")
pickle.dump(reducedY,pickle_out)
pickle_out.close()

pickle_out = open("Cat2.pickle","wb")
pickle.dump(reducedCat,pickle_out)
pickle_out.close()


# In[ ]:



        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




