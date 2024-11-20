#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import time
from time import sleep

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


# In[217]:


cols = pd.read_csv('cards.csv', nrows=1).columns
df = pd.read_csv('cards.csv', usecols=cols, low_memory=False)
desiredColumns = ['power', 'toughness', 'type_line', 'keywords', 'name','oracle_text','image_uris', 'multiverse_ids']
df.drop(df.columns.difference(desiredColumns), axis=1, inplace=True)
df = df[df['image_uris'].notna()]
df = df[df['name'].notna()]
df['image_uris'] = df.image_uris.apply(imageCleanup)
df['multiverse_ids'] = df.multiverse_ids.apply(idCleanup)
df['fileDownloadInfo'] = df.image_uris + ' E:/Jupyter/'+df.name.apply(nameCleaner) + "-" + df.multiverse_ids + ".png"
df.head()


# In[225]:


info = df['fileDownloadInfo']
count = 1
for i in info:
    
    fileName = i.split(" ")[1]
    img_data = requests.get(i.split(" ")[0]).content 
    newSize = 200
    image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    compressedImage = cv2.resize(image, (newSize, int(newSize*1.4)))
    print(fileName)
    cv2.imwrite(fileName, compressedImage)
    
    if( int(len(urls)/(5*count)) == i):
        count = 1 + count
        print("*",end="")
print("\n complete!")


# In[116]:





# In[5]:





# In[175]:





# In[223]:





# In[193]:





# In[ ]:




