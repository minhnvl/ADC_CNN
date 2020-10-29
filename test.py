import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
Classes = ["0.10%", "0.20%", "0.30%", "0.40%", "0.50%", "0.60%", "0.70%", "0.80%", "0.90%", "1%"]
df = pd.read_csv( './Dataset/Dataset_Training.csv',
                       header=None, low_memory=False
                       ).T
# df_2 = pd.read_csv("../input/ptbdb_abnormal.csv", header=None)
# df = pd.concat([df_1, df_2])
for idx, iclass in enumerate(Classes):
    df.loc[df.iloc[:,0] == iclass, 0] = idx
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


Y = np.array(df_train[0].values).astype(np.float)
X = np.array(df_train[1:].values)[..., np.newaxis].astype(np.float)

Y_test = np.array(df_test[0].values).astype(np.float)
X_test = np.array(df_test[1:].values)[..., np.newaxis].astype(np.float)
# print(Y)
# print(X)
# input()

def get_model():
    nclass = 1
    inp = Input(shape=(32768, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
file_path = "baseline_cnn_ptbdb_transfer_fullupdate.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
# model.load_weights("baseline_cnn_mitbih.h5", by_name=True)

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)
print(pred_test)
acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)












# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset
# import torchaudio
# import pandas as pd
# import numpy as np


# class UrbanSoundDataset(Dataset):
# #rapper for the UrbanSound8K dataset
#     # Argument List
#     #  path to the UrbanSound8K csv file
#     #  path to the UrbanSound8K audio files
#     #  list of folders to use in the dataset
    
#     def __init__(self, csv_path, file_path, folderList):
#         csvData = pd.read_csv(csv_path)
#         #initialize lists to hold file names, labels, and folder numbers
#         self.file_names = []
#         self.labels = []
#         self.folders = []
#         #loop through the csv entries and only add entries from folders in the folder list
#         for i in range(0,len(csvData)):
#             if csvData.iloc[i, 5] in folderList:
#                 self.file_names.append(csvData.iloc[i, 0])
#                 self.labels.append(csvData.iloc[i, 6])
#                 self.folders.append(csvData.iloc[i, 5])
                
#         self.file_path = file_path
#         # self.mixer = torchaudio.transforms.DownmixMono() #UrbanSound8K uses two channels, this will convert them to one
#         # self.mixer = torch.mean()
#         self.folderList = folderList
        
#     def __getitem__(self, index):
#         #format the file path and load the file
#         path = self.file_path + "fold" + str(self.folders[index]) + "/" + str(self.file_names[index]) + "/"
#         sound = torchaudio.load(path, out = None, normalization = True)
#         #load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
#         soundData = torch.mean(sound[0],dim=0, keepdim=True)
#         #downsample the audio to ~8kHz
#         tempData = torch.zeros([160000, 1]) #tempData accounts for audio clips that are too short
#         if soundData.numel() < 160000:
#             tempData[:soundData.numel()] = soundData[:]
#         else:
#             tempData[:] = soundData[:160000]
        
#         soundData = tempData
#         soundFormatted = torch.zeros([32000, 1])
#         soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
#         soundFormatted = soundFormatted.permute(1, 0)
#         return soundFormatted, self.labels[index]
    
#     def __len__(self):
#         return len(self.file_names)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(1, 128, 80, 4)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(128, 128, 3)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(128, 256, 3)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(256, 512, 3)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.pool4 = nn.MaxPool1d(4)
#         self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1
#         self.fc1 = nn.Linear(512, 10)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(self.bn3(x))
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(self.bn4(x))
#         x = self.pool4(x)
#         x = self.avgPool(x)
#         x = x.permute(0, 2, 1) #change the 512x1 to 1x512
#         x = self.fc1(x)
#         return F.log_softmax(x, dim = 2)

# def train(model, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         target = target.to(device)
#         data = data.requires_grad_() #set requires_grad to True for training
#         output = model(data)
#         output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10 
#         loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0: #print training stats
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss))
# def test(model, epoch):
#     model.eval()
#     correct = 0
#     for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         output = output.permute(1, 0, 2)
#         pred = output.max(2)[1] # get the index of the max log-probability
#         correct += pred.eq(target).cpu().sum().item()
#     print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#         correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
 
# csv_path = './Dataset/Dataset_Training.csv'
# file_path = './audio/'
# csvData = pd.read_csv(csv_path).T

# train_set = UrbanSoundDataset(csv_path, file_path, range(1,10))
# test_set = UrbanSoundDataset(csv_path, file_path, [10])

# # torch.cuda.set_device(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# print("Train set size: " + str(len(train_set)))
# print("Test set size: " + str(len(test_set)))

# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu

# train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)

# model = Net()
# model.to(device)
# print(model)  

# optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

# log_interval = 20
# for epoch in range(1, 41):
#     if epoch == 31:
#         print("First round of training complete. Setting learn rate to 0.001.")
#     scheduler.step()
#     train(model, epoch)
#     test(model, epoch)




