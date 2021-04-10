# -*- coding: utf-8 -*-
"""Point1.ipynb
#Implementation for Point 1
------
"""
# *** IMPORTANT ***
#If the notebook is being run on the local system, please download the necessary files from the drive #link provided in the code cells.

# *** IMPORTANT ***
#Unzip the file on the terminal using the below terminal:
#unzip train.zip

# Link to the zip file: https://drive.google.com/file/d/1g4dHphWLCX1PisdXacrfw8kdn7MGoxv5/view?usp=sharing

#Change path if necessary
#!cp -r "/content/drive/MyDrive/MIDAS/Point1/train.zip" "/content/train.zip" 



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
from torchvision import transforms
import shutil
from sklearn.metrics import auc, confusion_matrix, classification_report
import seaborn as sns
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import random

# Point 1's data is accessed via the zip file stored in the drive
# This zip file is transferred to the disk of the google colab, 
# because accessing it from the disk directly is faster than
# accessing the images from the drive

# Link to the zip file: https://drive.google.com/file/d/1g4dHphWLCX1PisdXacrfw8kdn7MGoxv5/view?usp=sharing

#Change path if necessary
#!cp -r "/content/drive/MyDrive/MIDAS/Point1/train.zip" "/content/train.zip" 

#Unzip the file on the terminal using the below terminal:
#unzip train.zip

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

#Used for augmenting training data
def _preprocess_aug(image):
  transform_aug = transforms.Compose([
     transforms.ToPILImage(),                 #Conversion to PIL Image                                      
     transforms.Resize((200, 200)),           #Resize image to 200 * 200                                                     
     transforms.RandomRotation(20),           #Apply rotations upto 20 degrees     
     #transforms.RandomAffine(0),
     transforms.ToTensor(),                   #Conversion to Tensor
     transforms.Normalize((0.5, ), (0.5,))    #Normalise Image
    ])

  return transform_aug(image)

"""Use the below class for augmentation"""

#Class used by DataLoader for training data
class Images_train(Dataset):
  def __init__(self, df):
        #DataFrame Structure must be of the form: | FilePath | Label |
        self.data = df

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    #print(index)
    image1 = cv.imread(self.data.iloc[index, 0], cv.IMREAD_GRAYSCALE) #Column 0: FilePath
    temp=random.randint(0,1)            #The temp variable takes care of the fact that not all images are augmented.
    if temp == 1:                       #If temp is 1, 
      image1 = _preprocess_aug(image1)  #augment the image
    else:                               #else
      image1 = _preprocess(image1)      #No augmentations are to be performed

    label = self.data.iloc[index, 1] #Column 1: Label

    return image1, torch.from_numpy(np.array([label], dtype=np.float32))

#Class used by DataLoader for testing data
class Images_test(Dataset):
  def __init__(self, df):
        self.data = df

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    #print(index)
    image1 = cv.imread(self.data.iloc[index, 0], cv.IMREAD_GRAYSCALE) #Column 0: FilePath
    image1 = _preprocess(image1)

    label = self.data.iloc[index, 1] #Column 1: Label

    return image1, torch.from_numpy(np.array([label], dtype=np.float32))



"""#Model Creation

The following is the description of the model architecture:
<font size=3px>
1. Conv(1, 32) -> ReLU -> Dropout -> Maxpool</br>
2. Conv(32, 64) -> ReLU -> Dropout -> Maxpool</br>
3. Conv(64, 128) -> ReLU -> Dropout -> Maxpool</br>
4. Conv(128, 256) -> ReLU -> Dropout -> Maxpool</br>
5. Conv(256, 512) -> ReLU -> Dropout -> Maxpool</br>
6. Transformer Encoder Layer</br>
7. Transformer Encoder Layer</br>
8. Transformer Encoder Layer</br>
9. Fully Connected Layer (512, 62)</br>
</font>

##Explanation:
The idea is to chain a convolutional neural network (CNN), with a Transformer encoder architecture. This CNN is responsible for extraction of the local information from the image and the Transformer encoder, reasons about the image as a whole and then generates the predictions. 

The transformer encoder is used to improve the embeddings recieved from the CNN, and the reason why it should succeed is because of the concept of "Attention" that is used in them. 



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

    self.zp3 = nn.ZeroPad2d(1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.mp3 = nn.MaxPool2d(kernel_size=2)

    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.mp4 = nn.MaxPool2d(kernel_size=2)

    self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.mp5 = nn.MaxPool2d(kernel_size=2)

    self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3) #num_layers=3: Three encoder layers in one block

    self.relu = nn.ReLU()

    self.row_emb = nn.Parameter(torch.rand(64, 512 // 2)) #[64, 256]
    self.col_emb = nn.Parameter(torch.rand(64, 512 // 2)) #[64, 256]

    self.fc4 = nn.Linear(512 * 6 * 6, 62)

    #Kindly ignore these layers
    self.fc1 = nn.Linear(512 * 6 * 6, 9216)
    self.fc2 = nn.Linear(9216, 4096)
    self.fc3 = nn.Linear(4096, 1024)


  def forward(self, x):
    x = self.dp(self.relu(self.conv1(x)))   #Conv -> ReLU -> Dropout: 200 * 200 * 32
    x = self.mp1(x)                         #MaxPool: 100 * 100 * 32

    x = self.dp(self.relu(self.conv2(x)))   #Conv -> ReLU -> Dropout: 100 * 100 * 64
    x = self.mp2(x)                         #MaxPool: 50 * 50 * 64

    x = self.dp(self.relu(self.conv3(x)))   #Conv -> ReLU -> Dropout: 50 * 50 * 128
    x = self.mp3(x)                         #MaxPool: 25 * 25 * 128

    x = self.dp(self.relu(self.conv4(x)))   #Conv -> ReLU -> Dropout: 25 * 25 * 256
    x = self.mp4(x)                         #MaxPool: 12 * 12 * 256

    x = self.dp(self.relu(self.conv5(x)))   #Conv -> ReLU -> Dropout: 12 * 12 * 512
    x = self.mp5(x)                         #MaxPool: 6 * 6 * 512

    H = x.shape[-1]#H = 6
    W = x.shape[-2]#W = 6

   
    pos = torch.cat([self.col_emb[:W].unsqueeze(0).repeat(H, 1, 1), self.row_emb[:H].unsqueeze(1).repeat(1, W, 1),], dim=-1).flatten(0, 1).unsqueeze(1)
    #Shape of pos: [6 * 6, 1,  512]

    x = x.flatten(2).permute(2, 0, 1)

    x = self.transformer_encoder(pos + x) 

    x = x.permute(1, 2, 0)
    x = torch.reshape(x, ((x.shape)[0], 512 * 36))

    #x = self.relu(self.fc1(x))
    #x = self.relu(self.fc2(x))
    #x = self.relu(self.fc3(x))
    x = self.fc4(x)
  
    return x


"""#Model Training"""

#Returns number of elements that are equal in the out (output) and labels (target) tensor
def accuracy(out, labels):
    count = 0
    _,pred = torch.max(out, dim=1)
    for i in range(out.shape[0]):
      if pred[i] == labels[i][0]:
        count = count + 1
    return count

"""With Augmentation"""
def train_eval():
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
        print(batch)
        
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

      tcorrect = tcorrect/(len(train_loader) * 8)
      vcorrect = vcorrect/len(val_loader)
      train_accuracy.append(tcorrect)
      valid_accuracy.append(vcorrect)

      # print training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
          epoch, train_loss, tcorrect, valid_loss, vcorrect))
      #print(train_loss_list)
      #print(valid_loss_list)

    chkpt_path = "/content/drive/MyDrive/MIDAS/Point1/model_encoder_aug.pt" 
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, chkpt_path)


def eval_model():
    #Evaluation begins
    i = 0
    l1 = []
    l2 = []

    model.eval()
    test_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    for batch, (img, target) in enumerate(test_loader):
        img, target = img.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.long)

        output = model(img)

        _,pred = torch.max(output, dim=1)
        l1.append(pred.item())
        l2.append(target.item)

    df = pd.DataFrame(columns=["Pred", "Target"])
    df["Pred"] = l1
    df["Target"] = l2

    y = np.array(df["Target"])
    y_pred = np.array(df["Pred"])

    print("Classification Report:")
    print(classification_report(y, y_pred))

#Add correct path of csv
df = pd.read_csv("path_to/MIDAS/ExecutableFiles/Point1/pointOne_cpath.csv")

#Add appropriate filepath of "Train" folder to existing filepath 
df['FilePath'] = "path_to/MIDAS/ExecutableFiles" + df['FilePath']
df = df.sample(frac = 1)

df_train, df_valid = train_test_split(df, test_size=0.25) #Split of 75/25
train_dataset = Images_train(df_train)
valid_dataset = Images_test(df_valid)

#Make sure that images are accessible via the path in the "FilePath" column
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  #Train
val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)  #Test

train_iter = iter(train_loader) 
images1, labels = train_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

val_iter = iter(val_loader)
images1, labels = val_iter.next()
print('images shape on batch size = {}'.format(images1.size()))
print('labels shape on batch size = {}'.format(labels.size()))

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

