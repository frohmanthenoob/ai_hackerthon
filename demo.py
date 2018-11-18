
# coding: utf-8

# In[1]:


from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import numpy as np
import ssl
import urllib
import cv2


# In[2]:


def getDf(path,W):
    o = {}
    o['name'] = []
    o['image'] = []
    for p in os.listdir(path):
        for filename in os.listdir(path+"/"+p):
            o['name'].append(p)
            oriimg = plt.imread(path+"/"+p+"/"+filename)
            height, width, depth = oriimg.shape
            imgScale = W/width if width>height else W/height
            newX,newY = width*imgScale, height*imgScale
            newSize = newX if newX>newY else newY
            newimg = cv2.resize(oriimg,(int(newSize),int(newSize)))
            o['image'].append(newimg.tolist())
    return o


# In[3]:


def imgResize(url,W):
    context = ssl._create_unverified_context()
    resp =  urllib.request.urlopen(url, context=context)
    oriimg = np.asarray(bytearray(resp.read()), dtype="uint8")
    oriimg = cv2.imdecode(oriimg, cv2.IMREAD_ANYCOLOR)
    oriimg = cv2.cvtColor(oriimg,cv2.COLOR_BGR2RGB)
    height, width, depth = oriimg.shape
    imgScale = W/width if width>height else W/height
    newX,newY = width*imgScale, height*imgScale
    newSize = newX if newX>newY else newY
    newimg = cv2.resize(oriimg,(int(newSize),int(newSize)))
    return np.array(newimg).reshape(-1,int(newSize),int(newSize),3)


# In[4]:


def getName(m,n,img):
    return pd.Categorical(n).categories[np.argmax(m.predict(img))]


# In[ ]:


if __name__ == '__main__':
    model = load_model('./after_hackerthon_ss.model')
    data = pd.DataFrame.from_dict(getDf('./data',250.))
    image = imgResize("https://diag.tactri.gov.tw/public/UploadDgnsResultImage/6519/2d8e4ed2-833c-433c-ac7d-3401af085a78.jpg",250.)
    print(getName(model,data['name'],image))

