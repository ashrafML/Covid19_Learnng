# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:13:42 2020

@author: a.ragab
"""#load lib
import os
import cv2
from imutils import paths
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import shutil
#load train text 
datatxt=pd.read_csv('test_split.txt',delimiter=' ',header=None)
datatxt=datatxt.rename(columns={0:'A',1:'B',2:'C'})

Covid_19=datatxt[datatxt['C']=='COVID-19']
#load imges
print("[INFO] loading images...")
os.path
imagePaths = list(paths.list_images("test"))
imagePaths44 = list(paths.list_images("Normal"))
data = []
labels = [] 
# loop over the image paths
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    for cvd in Covid_19.B:
        if cvd.lower()==imagePath.split('\\')[1].lower():
           test_image = Image.open(imagePath)
           test_image.save('Covid_19/'+imagePath.split('\\')[1])
            #cv2.imwrite('D:\courses\Ml Projects\Covid 19\Covid_19',image)   
           # im1.save('D:\courses\Ml Projects\Covid 19\Covid_19'+image)
#read real csv
           
dfcsv=pd.read_csv('metadata.csv')

Covid_19=dfcsv[dfcsv['finding']=='COVID-19']
data = []
labels = [] 
imagePaths = list(paths.list_images("images"))
# loop over the image paths
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    for cvd in Covid_19.filename:
        if cvd.lower()==imagePath.split('\\')[1].lower():
           test_image = Image.open(imagePath)
           test_image.save('metadataCovid/'+imagePath.split('\\')[1])