# -*- coding: utf-8 -*-
"""
#Dataset Generation

This notebook contains the code to generate a datatset using the patterns in folder structure.

If the notebook is being run on the local system, please download the necessary files from the drive link provided in the code cells.</br>
<font color="blue">*Please change the links accordingly*</font>
"""
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from collections import Counter 
import random

# Point 1's data is accessed via the zip file stored in the drive
# This zip file is transferred to the disk of the google colab, 
# because accessing it from the disk directly is faster than
# accessing the images from the drive

# Link to the zip file: https://drive.google.com/file/d/1g4dHphWLCX1PisdXacrfw8kdn7MGoxv5/view?usp=sharing

#Change path is necessary
#!cp -r "/content/drive/MyDrive/MIDAS/Point1/train.zip" "/content/train.zip" 

#Unzip file at terminal using the following command:
#unzip train.zip

path = "path_to_folder/MIDAS/ExecutableFiles/train"

def createData():
    df = pd.DataFrame(columns=['FilePath', 'Label'])
    i = 0
    for (root,dirs,files) in os.walk(path, topdown=True):
        if i == 0:
          i = 1
          continue

        #print(root)
        files_path = [root + '/' + i for i in files] #Appropriate path of root to be added before the image filepath
        label = [int(root[-2:]) - 1] * len(files_path) #Sample042 -> 42

        x = pd.DataFrame(list(zip(files_path, label)), columns=['FilePath', 'Label'])
        df = df.append(x)
        
    print(df.head())
    print(df.Label.value_counts())
    df.to_csv("pointOne_cpath.csv", index=None)


    """#Point 2

    The folders taken into consideration must be 001 to 009
    """
    df = pd.DataFrame(columns=['FilePath', 'Label'])
    lt = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']

    check = 0
    for (root,dirs,files) in os.walk(path, topdown=True):
        if check == 0:
          check = 1
          continue

        for i in lt:
          if i in root:
            #print(root)
            files_path = [path + root[root.find('/train'):] + '/' + i for i in files]
            label = [int(root[-2:]) - 1] * len(files_path)

            x = pd.DataFrame(list(zip(files_path, label)), columns=['FilePath', 'Label'])
            df = df.append(x)

    print(df.head())
    print(df.Label.value_counts())
    df.to_csv("one_to_ten.csv", index=None)



if __name__ == '__main__':
    createData()