# Point 2

#### 1. [Point2_Pretraining.ipynb](./Point2_Pretraining.ipynb): This file is used to train the model on 0-9 labelled images from [train.zip](../train.zip). Add the appropriate filepath to dataframes and change the filepath of where the model is to be stored. Make sure that the train.zip folder is unzipped. Filepath for images:

    “filepath of folder where train.zip is unzipped” + dataframe[“Filepath”]

#### 2. [Point2_Finetuning.ipynb](./Point2_Finetuning.ipynb): This file is used to retrain the model on MNIST. Change the filepath of where the model is to be stored.

#### 3. [Point2_Train_Scratch.ipynb](./Point2_Train_Scratch.ipynb): This file is used to train the model from scratch on MNIST. Change the filepath of where the model is to be stored.

#### 4. [one_to_ten.csv](./one_to_ten.csv): Has the filepath and labels of images labelled 0-9 from train.zip Has the image location as: “/train/….”. **Please do not modify this csv.**
