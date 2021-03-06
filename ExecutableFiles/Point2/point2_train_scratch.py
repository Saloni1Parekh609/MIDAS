# -*- coding: utf-8 -*-
"""Point2_Train_Scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W_n2OE5CFStZ0_nExiW_CekZrHjYcxEO

#Implementation of Point 2
------
"""
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


#All images are resized to 200 * 200 and have been normalized
img_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])

#Please change below links accordingly
trainset = datasets.MNIST('path_to/MIDAS/Point2', download=True, train=True, transform=img_transform)
testset = datasets.MNIST('path_to/MIDAS/Point2', download=True, train=False, transform=img_transform)

train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000]) #Training set is split into training and validation data

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

train_iter = iter(train_loader)
images1, labels = train_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

val_iter = iter(val_loader)
images1, labels = val_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

"""#Model Creation

<font size=3px>
1. Conv(1, 32) -> ReLU -> Dropout -> Maxpool</br>
2. Conv(32, 64) -> ReLU -> Dropout -> Maxpool</br>
3. Conv(64, 128) -> ReLU -> Dropout -> Maxpool</br>
4. Conv(128, 256) -> ReLU -> Dropout -> Maxpool</br>
5. Conv(256, 512) -> ReLU -> Dropout -> Maxpool</br>
6. Transformer Encoder Layer</br>
7. Transformer Encoder Layer</br>
8. Transformer Encoder Layer</br>
9. Fully Connected Layer (512, 10)</br>
</font>
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
    self.fc4 = nn.Linear(512 * 6 * 6, 10)

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
print(model)

"""#Model Training"""

#Returns number of elements that are equal in the out (output) and labels (target) tensor
def accuracy(out, labels):
    count = 0
    _,pred = torch.max(out, dim=1)
    for i in range(out.shape[0]):
      if pred[i] == labels[i]:
        count = count + 1
    return count

def train_eval():
    chkpt_path = "/content/drive/MyDrive/MIDAS/Point2/trained_scratch_model.pt"

    train_loss_list = []
    valid_loss_list = []
    train_accuracy = []
    valid_accuracy = []

    epochs = 10
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-4)
    criterion = nn.CrossEntropyLoss()
    min_valid_loss = np.Inf
    check_epoch = 5
    epoch_no_improve = 0
    i = 0


    for epoch in range(1, 31):
      train_loss = 0
      valid_loss = 0
      tcorrect = 0
      vcorrect = 0
      model.train()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("Training: ")
      for batch, (img, target) in enumerate(train_loader):
        #print(batch)
        img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target.view(-1))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)



      model.eval()
      print("Validation: ")
      for batch, (img, target) in enumerate(val_loader):
        img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        loss = criterion(output, target.view(-1))

        valid_loss += loss.item()
        vcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)


      # calculate average losses
      train_loss = train_loss/len(train_loader)
      valid_loss = valid_loss/len(val_loader)
      train_loss_list.append(train_loss)
      valid_loss_list.append(valid_loss)

      tcorrect = tcorrect/(len(train_loader) * 64)
      vcorrect = vcorrect/(len(val_loader) * 64)
      train_accuracy.append(tcorrect)
      valid_accuracy.append(vcorrect)

      # print training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
          epoch, train_loss, tcorrect, valid_loss, vcorrect))
      #print(train_loss_list)
      #print(valid_loss_list)

      if epoch % 5 == 0 :
        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
                  }, chkpt_path)

"""#Visualisation and Evaluation"""
def eval_model():
    #Evaluation begins
    i = 0
    l1 = []
    l2 = []
    vcorrect = 0

    model.eval()
    #test_loader = DataLoader(valset, batch_size=1, shuffle=True)
    for batch, (img, target) in enumerate(test_loader):
        img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        vcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)
        l1.append(pred.tolist())
        l2.append(target.tolist())

    vcorrect = vcorrect/(len(val_loader) * 64)
    df = pd.DataFrame(columns=["Pred", "Target"])
    l1 = [j for sub in l1 for j in sub]
    l2 = [j for sub in l2 for j in sub]
    df["Pred"] = l1
    df["Target"] = l2

    y = np.array(df["Target"])
    y_pred = np.array(df["Pred"])

    """**Evaluation**<br>
    At epoch 20:</br>
    Training Accuracy: 99.64% 	
    Validation Accuracy: 98.71%</br>
    Test Accuracy: 99%
    """

    print("Classification Report:")
    print(classification_report(y, y_pred))

if __name__ == '__main__':
    print("Data Loading...")
    print("Data Loaded...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smallModel().to(device)
    print(model)
    print("Model Created...")
    
    print("Model Training...")
    train_eval()    
    
    print("Model Evaluating...")
    eval_model()