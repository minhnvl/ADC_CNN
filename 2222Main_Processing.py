import os
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import random
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.metrics import r2_score
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import LambdaLR,StepLR

def Prepare_Dataset(path_txt,size,path_dataset):
    List_txt = [f for f in os.listdir(path_txt) if f.endswith(".csv")]
    size_train = int(len(List_txt)*size)
    random.shuffle(List_txt)
    List_Training = List_txt[:size_train]
    List_Testing = List_txt[size_train:]
    DF_Dataset_Training = pd.DataFrame()
    DF_Dataset_Testing = pd.DataFrame()

    for itxt in List_Training:
        print(itxt)
        datatxt = path_txt + itxt
        # data = open(datatxt, "r").read()
        DF_Data = pd.read_csv(datatxt)
        DF_Data = DF_Data.iloc[:,1:].sub(DF_Data.iloc[:,0].values,axis='rows')
        # DF_Data = DF_Data.iloc[:,1:]
        DF_Dataset_Training = pd.concat([DF_Dataset_Training,DF_Data],axis=1, sort=False)
        
    
    for itxt in List_Testing:
        print(itxt)
        datatxt = path_txt + itxt
        DF_Data = pd.read_csv(datatxt)
        DF_Data = DF_Data.iloc[:,1:].sub(DF_Data.iloc[:,0].values,axis='rows')
        # DF_Data = DF_Data.iloc[:,1:]
        DF_Dataset_Testing = pd.concat([DF_Dataset_Testing,DF_Data],axis=1, sort=False)
        
    DF_Dataset_Training.to_csv(path_dataset + "Dataset_Training.csv", index=False)
    DF_Dataset_Testing.to_csv(path_dataset + "Dataset_Testing.csv", index=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=1, padding=1)
        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(1,3)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(116512, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10) 
        
    def forward(self,x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = out.view(out.size(0), -1)
       
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        
        return out

class AlexNet_2D(nn.Module):
    def __init__(self):
        super(AlexNet_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(64,3), stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8,1)),

            nn.Conv2d(32, 64, kernel_size=(32,3), padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8,1)),

            nn.Conv2d(64, 128, kernel_size=(16,3), padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=(8,3), padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=(8,3), padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            
            # nn.Conv2d(16, 16, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(3,1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4864, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(256, 256),
            # nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.classifier(x)
    
        return x       




class AlexNet_1D(nn.Module):
    def __init__(self):
        super(AlexNet_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=5, padding=1),
            # nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),

            nn.Conv1d(16, 32, kernel_size=8, stride=3,padding=1),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6),

            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=1),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            

            nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=1),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            
            nn.Conv1d(128, 192, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(192, 192, kernel_size=2),
            nn.ReLU(inplace=True),


            # nn.Conv1d(64, 64, kernel_size=5, padding=1),
            # nn.ReLU(inplace=True),
            
            
            # nn.Conv1d(256, 512, kernel_size=4, padding=1),
            # nn.ReLU(inplace=True),
            
            # nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(768, 512),
            # nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 10),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 256),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, 10),
        )
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.classifier(x)
    
        return x 

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr
def Main_ADC():
    print("Training...")

    PATH_Model = './Model_ADC/'
    test_size = 0.3
    batch_size = 300
    slide_epoch = 100
    lr = 1e-3         # learning rate
    w_d = 1e-5        # weight decay
    momentum = 0.9   
    epochs = 500
    loss = 0
    # metrics = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(1)
    model = AlexNet_1D()
    model.to(device)
    m = nn.Sigmoid()
    if loss == 0:
        criterion = nn.CrossEntropyLoss().cuda()
    elif loss == 1:
        criterion = nn.MSELoss(reduction='mean').cuda()
        # criterion = nn.MSELoss()
    elif loss == 2:
        criterion = RMSELoss()
    elif loss == 3:
        criterion = nn.L1Loss().cuda()
    
 


    # train_set = Train_Loader()
    DF_Train = pd.read_csv( './Dataset/Dataset_Training.csv',
                       header=None, low_memory=False
                       ).T
   
    Classes = ["0.1%", "0.2%", "0.3%", "0.4%", "0.5%", "0.6%", "0.7%", "0.8%", "0.9%", "1%"]
    # print(DF_Train)
    for idx, iclass in enumerate(Classes):
        DF_Train.loc[DF_Train.iloc[:,0] == iclass, 0] = idx
    
    X = DF_Train.iloc[:,1:].astype(int).values
    Y = DF_Train.iloc[:,0].astype(int).values

    # scaler = MinMaxScaler() 
    # X = scaler.fit_transform(X) 
    
    features_train, features_test, targets_train, targets_test = train_test_split(X,Y,test_size=test_size, random_state=42)
 
    X_train = torch.from_numpy(features_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(features_test).type(torch.FloatTensor)

    Y_train = torch.from_numpy(targets_train).type(torch.LongTensor)
    Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)
    
    train_set = torch.utils.data.TensorDataset(X_train,Y_train)
    test_set = torch.utils.data.TensorDataset(X_test,Y_test)
 
    train_ = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    
    test_ = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        # drop_last=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_d)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    # model.train()
    best_acc = 0.0
    List_loss_train = []
    List_loss_test = []
    List_accuracy_train = []
    List_accuracy_test = []
    for epoch in range(epochs):
        adjust_learning_rate(lr,optimizer,epoch,slide_epoch)
        
        Val_loss_train, Accuracy_train, accuracy_training = Training_ADC(model,train_,train_set,device,adjust_learning_rate,loss,optimizer,criterion)
        Val_loss_test, Accuracy_test, accuracy_testing = Testing_ADC(model,test_,test_set,device,loss,criterion)

        List_loss_train.append(Val_loss_train)
        List_accuracy_train.append(Accuracy_train) 
        List_loss_test.append(Val_loss_test)
        List_accuracy_test.append(Accuracy_test)

        print('-----------------------------------------------')
        print("Epoch: {}/{}\t ".format(epoch+1, epochs),
              "Training Loss: {:.3f}\t ".format(Val_loss_train),
              "Testing Loss: {:.3f}\t ".format(Val_loss_test),
              "Training Accuracy: {:.1f}% {}/{}\t".format(Accuracy_train,accuracy_training.item(),len(train_set)),
              "Testing Accuracy: {:.1f}% {}/{}".format(Accuracy_test, accuracy_testing.item(), len(test_set))
            )
        
        if Accuracy_test > best_acc:
            best_acc = Accuracy_test
            print('==>>>Saving model ...')
            if not os.path.exists(PATH_Model):
                os.makedirs(PATH_Model)
            torch.save(model.state_dict(), '%s/Model_ADC_%03d.pth' % (PATH_Model ,epoch))
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Accuracy and Loss')
    axs[0].plot(List_accuracy_train,label = "Training")
    axs[0].plot(List_accuracy_test,label = "Testing")
    axs[0].legend(loc='lower right')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_ylim([0,100])
    axs[1].plot(List_loss_train,label = "Training")
    axs[1].plot(List_loss_test,label = "Testing")
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    plt.show()

def Training_ADC(model,train_,train_set,device,adjust_learning_rate,loss,optimizer,criterion):
    train_loss = 0.0
    accuracy_training = 0
    model.train()
    for datatrain, label in train_:
        # datatrain = Variable(datatrain.view(datatrain.size(0),1,datatrain.size(1),1)).to(device)
        datatrain = Variable(datatrain.view(datatrain.size(0),1,datatrain.size(1))).to(device)
        label = Variable(label).to(device)
        output_train = model(datatrain)
        optimizer.zero_grad()
        
        
        if loss == 0:
            loss_train = criterion(output_train, label)
        elif loss == 1:
            loss_train = criterion(output_train[:, -1], label.type(torch.FloatTensor).cuda())
        elif loss == 2:
            loss_train = criterion(output_train[:, -1], label.type(torch.FloatTensor).cuda())
        elif loss == 3:
            loss_train = criterion(output_train[:, -1], label.type(torch.FloatTensor).cuda())
      
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        
        pred = output_train.data.max(1,keepdim=True)[1]
        label = label.type(torch.LongTensor)
        accuracy_training += pred.eq(label.data.view_as(pred).to(device)).sum()

    Val_loss_train = train_loss/len(train_set)
    Accuracy_train = 100. * accuracy_training.item()/len(train_set)
    
    return Val_loss_train, Accuracy_train, accuracy_training

def Testing_ADC(model,test_,test_set,device,loss,criterion):
    test_loss = 0
    accuracy_testing = 0
    with torch.no_grad(): #Turning off gradients to speed up
        
        model.eval()
        
        for datatest,label in test_:
            # datatest = Variable(datatest.view(datatest.size(0),1,datatest.size(1),1)).to(device)
            datatest = Variable(datatest.view(datatest.size(0),1,datatest.size(1))).to(device)
            label = Variable(label).to(device)
            output_test = model(datatest)

            if loss == 0:
                loss_test = criterion(output_test, label)
            elif loss == 1:
                loss_test = criterion(output_test[:, -1], label.type(torch.FloatTensor).cuda())
            elif loss == 2:
                loss_test = criterion(output_test[:, -1], label.type(torch.FloatTensor).cuda())
            elif loss == 3:
                loss_test = criterion(output_test[:, -1], label.type(torch.FloatTensor).cuda())
        
            test_loss += loss_test
            pred = output_test.data.max(1,keepdim=True)[1]
            accuracy_testing += pred.eq(label.type(torch.LongTensor).data.view_as(pred).to(device)).sum()


        Val_loss_test = test_loss/len(test_set)
        Accuracy_test = 100. * accuracy_testing.item()/len(test_set)
    return Val_loss_test, Accuracy_test, accuracy_testing

if __name__ == "__main__":
    path_txt = "./Inputtxt/"
    path_dataset = "./Dataset/"
    size = 0.8
    # Prepare_Dataset(path_txt,size,path_dataset)
    Main_ADC()
