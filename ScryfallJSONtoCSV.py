#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import csv


# In[9]:


f = open("cards.json", encoding="utf8")
data = json.load(f)
biggestKey = 0
MasterKey = []
for cards in data:
    if biggestKey < len(list(cards.keys())):
        biggestKey = len(list(cards.keys()))
        MasterKey = list(cards.keys())
data_file = open('cards.csv', 'w', newline='', encoding="utf-8")

csv_writer = csv.writer(data_file)
csv_writer.writerow(MasterKey)
for cards in data:
    if list(cards.values())[3]:
        organizedData = []
        localTypes = list(cards.keys())
        localValues = list(cards.values())
        for Mastertypes in MasterKey:
            found = False
            for i in range(len(localTypes)):
                if(Mastertypes == localTypes[i]):
                    organizedData.append(localValues[i])
                    localTypes.pop(i)
                    localValues.pop(i)
                    found = True
                    break
            if not found:
                    organizedData.append(None)
        csv_writer.writerow(organizedData)

data_file.close()


# In[ ]:




