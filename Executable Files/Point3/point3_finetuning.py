# -*- coding: utf-8 -*-
"""Point3_Finetuning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J5_DcgK4IqupfT0j0HlOIsIbE_sPYVzt

#Implementation of Point 3
------
"""

# *** IMPORTANT ***
#If the notebook is being run on the local system, please download the necessary files from the drive #link provided in the code cells.

# *** IMPORTANT ***
#Unzip the file on the terminal using the below terminal:
#unzip mnistTask.zip

# Link to the zip file: https://drive.google.com/file/d/1bWjl661Kk3xa8MmdXKHAwsr48080LaQG/view?usp=sharing

#Change path if necessary
#cp -r "/content/drive/MyDrive/MIDAS/Point3/mnistTask.zip" "/content/mnistTask.zip"

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
#from pycm import *

"""#Data Preparation"""

def _preprocess(image):
    # Preprocssing step
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])
    return img_transform(image)

def _preprocess_aug(image):
  transform_aug = transforms.Compose([
     transforms.ToPILImage(),                                     
     transforms.Resize((200, 200)),                                                        
     transforms.RandomRotation(20),
     #transforms.RandomAffine(0),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5,))
    ])

  return transform_aug(image)

class Images_train(Dataset):
  def __init__(self, df):
        self.data = df

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    #print(index)
    image1 = cv.imread(self.data.iloc[index, 0], cv.IMREAD_GRAYSCALE)
    temp=random.randint(0,1)
    if temp == 1:
      image1 = _preprocess_aug(image1)
    else:
      image1 = _preprocess(image1)

    label = self.data.iloc[index, 1]

    return image1, torch.from_numpy(np.array([label], dtype=np.float32))

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

    return image1, torch.from_numpy(np.array([label], dtype=np.float32))


img_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5,))
    ])


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

"""#Loading the Model"""

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

print(model.parameters)

"""For successful finetuning, reset the last layer only, such that instead of 62 the model only predicts 10 classes."""

num_ftrs = model.fc4.in_features
model.fc4 = nn.Linear(num_ftrs, 10)

model = model.cuda()

"""#Model Training"""

#Returns number of elements that are equal in the out (output) and labels (target) tensor
def accuracy(out, labels):
    count = 0
    _,pred = torch.max(out, dim=1)
    for i in range(out.shape[0]):
      if pred[i] == labels[i][0]:
        count = count + 1
    return count

#34 mins
def train_eval():
    chkpt_path = "/content/drive/MyDrive/MIDAS/Point3/finetuned_model.pt"

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


    for epoch in range(1, 30):
      train_loss = 0
      valid_loss = 0
      tcorrect = 0
      vcorrect = 0
      model.train()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print("Training: ")
      for batch, (img, target) in enumerate(train_loader):
        #print(batch)
        if torch.cuda.is_available():
            img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target.view(-1))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)
        """print('Pred: ')
        print(pred)
        print('Target: ')
        print(target)"""


      model.eval()
      print("Validation: ")
      for batch, (img, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        loss = criterion(output, target.view(-1))

        valid_loss += loss.item()
        vcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)
        """print('Pred: ')
        print(pred)
        print('Target: ')
        print(target)"""

      # calculate average losses
      train_loss = train_loss/len(train_loader)
      valid_loss = valid_loss/len(val_loader)
      train_loss_list.append(train_loss)
      valid_loss_list.append(valid_loss)

      tcorrect = tcorrect/len(train_dataset)
      vcorrect = vcorrect/len(valid_dataset)
      train_accuracy.append(tcorrect)
      valid_accuracy.append(vcorrect)

      # print training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
          epoch, train_loss, tcorrect, valid_loss, vcorrect))
      #print(train_loss_list)
      #print(valid_loss_list)

      #print(valid_loss_list)

      if epoch % 5 == 0 :
        torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': loss,
                  }, chkpt_path)


def model_eval():
    #Evaluation begins
    i = 0
    l1 = []
    l2 = []

    model.eval()
    #test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    for batch, (img, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        #vcorrect += accuracy(output, target)
        _,pred = torch.max(output, dim=1)
        l1.append(pred.item())
        l2.append(target.item())

    df = pd.DataFrame(columns=["Pred", "Target"])
    df["Pred"] = l1
    df["Target"] = l2

    y = np.array(df["Target"])
    y_pred = np.array(df["Pred"])

    print("Classification Report:")
    print(classification_report(y, y_pred))

"""#Evaluation on MNIST's Test Set"""

def accuracy_test(out, labels):
    count = 0
    _,pred = torch.max(out, dim=1)
    for i in range(out.shape[0]):
      if pred[i] == labels[i]:
        count = count + 1
    return count

def eval_mnist():
    #Evaluation begins
    i = 0
    l1 = []
    l2 = []
    vcorrect = 0

    model.eval()
    #test_loader = DataLoader(valset, batch_size=1, shuffle=True)
    for batch, (img, target) in enumerate(test_loader):
        if torch.cuda.is_available():
            img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        vcorrect += accuracy_test(output, target)
        _,pred = torch.max(output, dim=1)
        l1.append(pred.tolist())
        l2.append(target.tolist())

    #vcorrect = vcorrect/(len(val_loader) * 64)
    df = pd.DataFrame(columns=["Pred", "Target"])
    l1 = [j for sub in l1 for j in sub]
    l2 = [j for sub in l2 for j in sub]
    df["Pred"] = l1
    df["Target"] = l2
    df.head()

    y = np.array(df["Target"])
    y_pred = np.array(df["Pred"])

    """**Evaluation**<br>
    At epoch 20:</br>
    Training Accuracy: 99.06% 	
    Validation Accuracy: 99.07%</br>
    Test Accuracy: 99%
    """

    print("Classification Report:")
    print(classification_report(y, y_pred))

df = pd.read_csv("/content/drive/MyDrive/MIDAS/Point3/dataset_p3.csv")
#Add appropriate filepath of "Train" folder to existing filepath 
df['FilePath'] = "path_to/MIDAS/" + df['FilePath']
df = df.sample(frac = 1)

df_train, df_valid = train_test_split(df, test_size=0.25)

train_dataset = Images_train(df_train)
valid_dataset = Images_test(df_valid)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

train_iter = iter(train_loader)
images1, labels = train_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

val_iter = iter(val_loader)
images1, labels = val_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

testset = datasets.MNIST('/content/drive/MyDrive/MIDAS/Point2/MNIST', download=True, train=False, transform=img_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

val_iter = iter(test_loader)
images1, labels = val_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

if __name__ == '__main__':
    print("Data Loading...")
    print("Data Loaded...")
    
    print("Model Created...")
    
    print("Model Training...")
    train_eval()    
    
    print("Model Evaluating...")
    eval_model()
    
    print("MNIST Evaluation...")
    eval_mnist()



