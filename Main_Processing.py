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
import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='Convolution Neural Network for ADC')
    parser.add_argument('--path_dataset',type=str,default="./Dataset/",metavar='N',
                        help='Path Dataset for Pre-Processing data')
    parser.add_argument('--path_csv',type=str,default="./InputCsv/",metavar='N',
                        help='Path CSV for Pre-Processing data')
    parser.add_argument('--path_model',type=str,default="./Model_ADC/",metavar='N',
                        help='Path Model')
    parser.add_argument('--predata',type=int,default=0,metavar='N',
                        help='Pre-processing Data or Training (default: training)')
    parser.add_argument('--data_size',type=float,default=0.8,metavar='M',
                        help='Data size for Pre-Processing data')
    parser.add_argument('--test_size',type=float,default=0.2,metavar='M',
                        help='Test size for Training')
    parser.add_argument('--batch-size',type=int,default=300,metavar='N',
                        help='batch size for training(default: 300)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=500,metavar='N',
                        help='number of epoch to train(default: 500)')
    parser.add_argument('--lr_epoch',type=int,default=75,metavar='N',
                        help='number of epochs to decay learning rate(default: 75)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=0,metavar='S',
                        help='random seed(default: 0)')

    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def Prepare_Dataset(args):
    path_Csv,path_dataset,size = args.path_csv, args.path_dataset, args.data_size
    List_Csv = [f for f in os.listdir(path_Csv) if f.endswith(".csv")]
    size_train = int(len(List_Csv)*size)
    random.shuffle(List_Csv)
    List_Training = List_Csv[:size_train]
    List_Testing = List_Csv[size_train:]
    DF_Dataset_Training = pd.DataFrame()
    DF_Dataset_Testing_sub = pd.DataFrame()
    DF_Dataset_Testing_org = pd.DataFrame()
    LST_hearder = ['ideal', '0.1%', '0.2%', '0.3%', '0.4%', '0.5%', '0.6%', '0.7%', '0.8%', '0.9%', '1%']
    for itxt in List_Training:
        print("Training File: %s"%itxt)
        datatxt = path_Csv + itxt
        DF_Data = pd.read_csv(datatxt)
        DF_Data.columns = LST_hearder
        DF_Data_sub = DF_Data.iloc[:,1:].sub(DF_Data.iloc[:,0].values,axis='rows')
        DF_Dataset_Training = pd.concat([DF_Dataset_Training,DF_Data_sub],axis=1, sort=False)
        
    
    for itxt in List_Testing:
        print("Testing File: %s"%itxt)
        datatxt = path_Csv + itxt
        DF_Data = pd.read_csv(datatxt)
        DF_Data.columns = LST_hearder
        DF_Data_sub = DF_Data.iloc[:,1:].sub(DF_Data.iloc[:,0].values,axis='rows')
        DF_Dataset_Testing_sub = pd.concat([DF_Dataset_Testing_sub,DF_Data_sub],axis=1, sort=False)
        DF_Dataset_Testing_org = pd.concat([DF_Dataset_Testing_org,DF_Data],axis=1, sort=False)

    DF_Dataset_Training.to_csv(path_dataset + "Dataset_Training.csv", index=False)
    DF_Dataset_Testing_sub.to_csv(path_dataset + "Dataset_Testing.csv", index=False)
    DF_Dataset_Testing_org.to_csv(path_dataset + "Dataset_Testing_org.csv", index=False)



class MinvlNet(nn.Module):
    def __init__(self):
        super(MinvlNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8),

            nn.Conv1d(16, 32, kernel_size=8, stride=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=6),

            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(192, 512, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv1d(512, 512, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 10),
      
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
def Main_ADC(args):
    print("Training...")
    Classes = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%', '0.6%', '0.7%', '0.8%', '0.9%', '1%']
    path_model, test_size, epochs, batch_size, slide_epoch, lr, device_id,w_d, momentum = args.path_model, args.test_size, \
                                                                                args.epochs, args.batch_size, \
                                                                                args.lr_epoch, args.lr, args.seed, \
                                                                                args.weight_decay, args.momentum
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device_id)
    model = MinvlNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
   
    DF_Train = pd.read_csv( './Dataset/Dataset_Training.csv',
                       header=None, low_memory=False
                       ).T
    print("Total value of Training: %d"%len(DF_Train))
    

    for idx, iclass in enumerate(Classes):
        DF_Train.loc[DF_Train.iloc[:,0] == iclass, 0] = idx
    
    X = DF_Train.iloc[:,1:].astype(int).values
    Y = DF_Train.iloc[:,0].astype(int).values

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
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )

    
    test_ = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        # drop_last=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_d)
    best_acc = 0.0
    List_loss_train = []
    List_loss_test = []
    List_accuracy_train = []
    List_accuracy_test = []
    for epoch in range(epochs):
        adjust_learning_rate(lr,optimizer,epoch,slide_epoch)
        
        Val_loss_train, Accuracy_train, accuracy_training = Training_ADC(model,train_,train_set,device,adjust_learning_rate,optimizer,criterion)
        Val_loss_test, Accuracy_test, accuracy_testing = Testing_ADC(model,test_,test_set,device,criterion)

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
            if not os.path.exists(path_model):
                os.makedirs(path_model)
            torch.save(model.state_dict(), '%s/Model_ADC_%03d.pth' % (path_model ,epoch))
    
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

def Training_ADC(model,train_,train_set,device,adjust_learning_rate,optimizer,criterion):
    train_loss = 0.0
    accuracy_training = 0
    model.train()
    for datatrain, label in train_:
        # datatrain = Variable(datatrain.view(datatrain.size(0),1,datatrain.size(1),1)).to(device)
        datatrain = Variable(datatrain.view(datatrain.size(0),1,datatrain.size(1))).to(device)
        label = Variable(label).to(device)
        output_train = model(datatrain)
        optimizer.zero_grad()
        loss_train = criterion(output_train, label)
        
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        
        pred = output_train.data.max(1,keepdim=True)[1]
        label = label.type(torch.LongTensor)
        accuracy_training += pred.eq(label.data.view_as(pred).to(device)).sum()

    Val_loss_train = train_loss/len(train_set)
    Accuracy_train = 100. * accuracy_training.item()/len(train_set)
    
    return Val_loss_train, Accuracy_train, accuracy_training

def Testing_ADC(model,test_,test_set,device,criterion):
    test_loss = 0
    accuracy_testing = 0
    with torch.no_grad(): #Turning off gradients to speed up
        model.eval()
        for datatest,label in test_:
            # datatest = Variable(datatest.view(datatest.size(0),1,datatest.size(1),1)).to(device)
            datatest = Variable(datatest.view(datatest.size(0),1,datatest.size(1))).to(device)
            label = Variable(label).to(device)
            output_test = model(datatest)
            loss_test = criterion(output_test, label)
          
            test_loss += loss_test
            pred = output_test.data.max(1,keepdim=True)[1]
            accuracy_testing += pred.eq(label.type(torch.LongTensor).data.view_as(pred).to(device)).sum()


        Val_loss_test = test_loss/len(test_set)
        Accuracy_test = 100. * accuracy_testing.item()/len(test_set)
    return Val_loss_test, Accuracy_test, accuracy_testing

if __name__ == "__main__":
    args = ParseArgs()
    if (args.predata == 1):
        Prepare_Dataset(args)
    else:
        Main_ADC(args)
