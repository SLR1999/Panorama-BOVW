#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import imutils
from imutils import paths
import random
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics


# In[ ]:


imagePaths = list(imutils.paths.list_images("./cifar-10/airplane")) + list(imutils.paths.list_images("./cifar-10/automobile")) + list(imutils.paths.list_images("./cifar-10/bird")) + list(imutils.paths.list_images("./cifar-10/cat")) + list(imutils.paths.list_images("./cifar-10/deer")) + list(imutils.paths.list_images("./cifar-10/dog")) + list(imutils.paths.list_images("./cifar-10/frog")) + list(imutils.paths.list_images("./cifar-10/horse")) +  list(imutils.paths.list_images("./cifar-10/ship"))  +  list(imutils.paths.list_images("./cifar-10/truck"))
   
labels = []
descriptorList = []

sift = cv2.xfeatures2d.SIFT_create()

# print ("here")

for (i, imagePath) in enumerate(imagePaths):
   image = cv2.imread(imagePath)
   gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   label = imagePath.split(os.path.sep)[2].split("/")[0]
   keyPoints, descriptor = sift.detectAndCompute(image, None)
   descriptorList.append((imagePath, descriptor))
#     print (imagePath)
   if (label == 'airplane'):
       labels.append(0)
   elif (label == 'automobile'):
       labels.append(1)
   elif (label == 'bird'):
       labels.append(2)
   elif (label == 'cat'):
       labels.append(3)
   elif (label == 'deer'):
       labels.append(4)
   elif (label == 'dog'):
       labels.append(5)
   elif (label == 'frog'):
       labels.append(6)
   elif (label == 'horse'):
       labels.append(7)
   elif (label == 'ship'):
       labels.append(8)
   else:
       labels.append(9)
       
# print (labels)
# print (descriptorList[0])
# print (descriptorList[10])
       
       
descriptors = descriptorList[0][1]
# print ((descriptors))
   
for imageFlag, descriptor in descriptorList[1:]:
   if descriptor is not None:
       descriptors = np.vstack((descriptors, descriptor))


# In[ ]:


k = 50

voc,  variance = kmeans(descriptors, k, 1)
print (variance)
print (len(voc))


# In[ ]:


imageFeatures = np.zeros((len(imagePaths), k), "float32")
for i in range(len(imagePaths)):
    words, distance = vq(descriptorList[i][1],voc)
    for w in words:
        imageFeatures[i][w] += 1


# In[ ]:


stdSlr = StandardScaler().fit(imageFeatures)
imageFeatures = stdSlr.transform(imageFeatures)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(imageFeatures, labels, test_size=0.16, random_state=4)


# In[ ]:


clf = cv2.ml.KNearest_create()
clf.train(X_train, cv2.ml.ROW_SAMPLE, np.asarray(y_train, dtype=np.float32))


# In[ ]:


ret, results, neighbours ,dist = clf.findNearest(X_test, k=10)


# In[ ]:


pred_label = []
for var in results:
    label = var
    pred_label.append(int(label))

print (y_test)
print (pred_label)
    
metrics.accuracy_score(y_test, pred_label)

