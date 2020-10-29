import pandas as pd
import Main_Processing as M
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
DF_Test = pd.read_csv( './Dataset/Dataset_Testing.csv',
                    header=None, low_memory=False
                    ).T
DF_Test_org = pd.read_csv( './Dataset/Dataset_Testing_org.csv',
                    header=None, low_memory=False
                    ).T

Classes = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%', '0.6%', '0.7%', '0.8%', '0.9%', '1%']

DF_Test = DF_Test.sample(frac=1).reset_index(drop=True)
DF_Test_org = DF_Test_org.sample(frac=1).reset_index(drop=True)

for idx, iclass in enumerate(Classes):
    DF_Test.loc[DF_Test.iloc[:,0] == iclass, 0] = idx
    DF_Test_org.loc[DF_Test_org.iloc[:,0] == iclass, 0] = idx

# print(DF_Test)


# X_test = torch.from_numpy(X).type(torch.FloatTensor)
# Y_test = torch.from_numpy(Y).type(torch.LongTensor)

# test_set = torch.utils.data.TensorDataset(X_test,Y_test)
PATH_Model = "./Model_ADC//Model_ADC_382.pth"

model = M.AlexNet_1D().cuda()
model.load_state_dict(torch.load(PATH_Model))
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(1)
model.to(device)

Lst_true = []
Lst_false = []
lst_nearby_result =[]

for index, DF in DF_Test.iterrows():
    
    X = np.array(DF[1:].astype(int))
    label = DF[0]
    X_test = torch.from_numpy(X).type(torch.FloatTensor)
    datatest = Variable(X_test.view(1,1,X_test.size(0))).to(device)
    
    
    outputs = model(datatest)
    _, predicted = torch.max(outputs.data, 1)

    if (int(predicted[0]) == label ):
        Lst_true.append(index)
    else:
        #A = np.array(DF_Test_org.iloc[index,1:].astype(int)).sum()
        #B = datatest.view(-1).sum()
        #C = np.array(DF_Test_org.iloc[index,1:].astype(int))
        #D = np.array(datatest.view(-1).cpu())
        #E = np.divide(D,C)
        #print(np.array(DF_Test_org.iloc[index,1:].astype(int)).sum())
        #print(datatest.view(-1).sum())
        #print(A*100/(A+B.item()))
        #print(B.item()*100/A)
        #print(E.sum())
        #print(Classes[int(predicted)])
        #print(Classes[label])
        #input()
        denta = max(label,predicted)/min(label,predicted)
        if denta <= 4:
            lst_nearby_result.append(index)
        Lst_false.append(index)
# print(Lst_true)
# print(Lst_false)
print("Case True: %d/%d\t%d " %(len(Lst_true),len(DF_Test), len(Lst_true)*100/len(DF_Test)))
print("Case False: %d/%d\t%d " %(len(Lst_false),len(DF_Test),len(Lst_false)*100/len(DF_Test)))
print("The number of nearly Result: %d"%(len(lst_nearby_result)))
print("The wrong result: %d"%(len(Lst_false)-len(lst_nearby_result)))
