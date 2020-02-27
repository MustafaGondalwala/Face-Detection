#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import sys


# In[19]:


imagePath = sys.argv[0]
cascPath = "haarcascade_frontalface_default.xml"


# In[11]:


faceCascade = cv2.CascadeClassifier(cascPath)


# In[12]:


image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[14]:


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.FONT_HERSHEY_SIMPLEX
)


# In[16]:


print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


# In[17]:


cv2.imshow("Faces found", image)
cv2.waitKey(0)


# In[ ]:




