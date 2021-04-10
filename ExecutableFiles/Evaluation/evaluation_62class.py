# -*- coding: utf-8 -*-
"""Evaluation_62class.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tD6TfUtrEuP4VYA-auX1LLSVXQWGaQ49

#Evaluation
-------

The following dataset can be evaluated:
CSV file with columns | FilePath | Label |

This is a 62-class classification task
"""

#Unzip a file if necessary

# Link to the zip file: https://drive.google.com/file/d/1g4dHphWLCX1PisdXacrfw8kdn7MGoxv5/view?usp=sharing
#!cp -r "/content/drive/MyDrive/MIDAS/Point1/train.zip" "/content/train.zip" 
#!unzip train.zip

import numpy as np
import torch
import csv
from torch import nn
import pandas as pd
import cv2 as cv
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ZeroPad2d
from torchvision import transforms, datasets
import shutil
from sklearn.metrics import auc, confusion_matrix, classification_report
import seaborn as sns
from scipy.optimize import brentq
from scipy.interpolate import interp1d 
import random
from pycm import *

"""#Data Preparation for CSVs


"""

#All images are resized to 200 * 200 and have been normalized
def _preprocess(image):
    # Preprocessing step
    img_transform = transforms.Compose([
        transforms.ToPILImage(),              #Conversion to PIL Image
        transforms.Resize((200, 200)),        #Resize image to 200 * 200
        transforms.ToTensor(),                #Conversion to Tensor
        transforms.Normalize((0.5, ), (0.5,)) #Normalise Image
    ])
    return img_transform(image)

class Images_test(Dataset):
  def __init__(self, df):
        self.data = df

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    #print(index)
    image1 = cv.imread(self.data.iloc[index, 0], cv.IMREAD_GRAYSCALE)
    image1 = _preprocess(image1)

    label = self.data.iloc[index, 1]

    return image1, torch.from_numpy(np.array([label], dtype=np.float32)), self.data.iloc[index, 0]

df = pd.read_csv("/content/drive/MyDrive//MIDAS/ExecutableFiles/Point1/pointOne_cpath.csv") #Please add appropriate link
df["FilePath"] = "path_to/MIDAS/ExecutableFiles" + df["FilePath"]
df = df.sample(frac = 1)
df.head()

valid_dataset = Images_test(df)
val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

val_iter = iter(val_loader)
images1, labels, fpath = val_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))
print('File path shape on batch size = {}'.format(fpath))

"""#Evaluation

"""

class smallModel(nn.Module):
  def __init__(self):
    super(smallModel, self).__init__()
    
    self.zp1 = nn.ZeroPad2d(1)
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.mp1 = nn.MaxPool2d(kernel_size=2)
    self.dp = nn.Dropout(p=0.3)

    self.zp2 = nn.ZeroPad2d(1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.mp2 = nn.MaxPool2d(kernel_size=2)
    #self.dp = nn.Dropout(p=0.3)

    self.zp3 = nn.ZeroPad2d(1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.mp3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.mp4 = nn.MaxPool2d(kernel_size=2)

    self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.mp5 = nn.MaxPool2d(kernel_size=2)

    self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

    self.relu = nn.ReLU()

    self.fc1 = nn.Linear(512 * 6 * 6, 9216)
    self.fc2 = nn.Linear(9216, 4096)
    self.fc3 = nn.Linear(4096, 1024)
    self.fc4 = nn.Linear(512 * 6 * 6, 62)

    self.row_emb = nn.Parameter(torch.rand(64, 512 // 2))
    self.col_emb = nn.Parameter(torch.rand(64, 512 // 2))

  def forward(self, x):
    x = self.dp(self.relu(self.conv1(x)))
    x = self.mp1(x)

    x = self.dp(self.relu(self.conv2(x)))
    x = self.mp2(x)

    x = self.dp(self.relu(self.conv3(x)))
    x = self.mp3(x)

    x = self.dp(self.relu(self.conv4(x)))
    x = self.mp4(x)

    x = self.dp(self.relu(self.conv5(x)))
    x = self.mp5(x)

    H = x.shape[-1]
    W = x.shape[-2]
    pos = torch.cat([self.col_emb[:W].unsqueeze(0).repeat(H, 1, 1), self.row_emb[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1).flatten(0, 1).unsqueeze(1)

    x = x.flatten(2).permute(2, 0, 1)

    x = self.transformer_encoder(pos + x)

    x = x.permute(1, 2, 0)
    x = torch.reshape(x, ((x.shape)[0], 512 * 36))

    #x = self.relu(self.fc1(x))
    #x = self.relu(self.fc2(x))
    #x = self.relu(self.fc3(x))
    x = self.fc4(x)
  
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smallModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4)

#Link to the model: https://drive.google.com/file/d/1TwFAjRqCk433H_Axh-PPms-yZwUuqlEE/view?usp=sharing
checkpoint = torch.load("/content/drive/MyDrive/MIDAS/Point1/model_encoder_aug.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_last = checkpoint['epoch']
loss = checkpoint['loss']

print(loss)
print(epoch_last)

def eval_model():
    #Evaluation begins
    i = 0
    l1 = []
    l2 = []
    vcorrect = 0

    model.eval()
    #test_loader = DataLoader(valset, batch_size=1, shuffle=True)
    for batch, (img, target, fpath) in enumerate(val_loader):
        img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        print(batch)
        output = model(img)

        _,pred = torch.max(output, dim=1)
        l1.append(target.item())
        l2.append(pred.item())

    df_label = pd.DataFrame(columns=["Target", "Pred"])
    df_label["Target"] = l1
    df_label["Pred"] = l2
    df_label.head()

    y = np.array(df_label["Target"])
    y_pred = np.array(df_label["Pred"])

    print("Classification Report:")
    print(classification_report(y, y_pred))
    
if __name__ == '__main__':
    print("Model Evaluating...")
    eval_model()

