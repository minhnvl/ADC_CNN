import pandas as pd
import Main_Processing as M
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import re
import math

Path_Input = "./Test_Input/TI_200M.csv" 
DF_Test = pd.read_csv( Path_Input,
                    header=None, low_memory=False
                    ).T
DF_Test = DF_Test.iloc[1,1:]
X = np.array(DF_Test.astype(int))

def Predict_TimeSkew():

    #print(DF_Test)
    Classes = ['0.1%', '0.2%', '0.3%', '0.4%', '0.5%', '0.6%', '0.7%', '0.8%', '0.9%', '1%']

    

    PATH_Model = "./Model_ADC//Model_ADC_382.pth"

    model = M.AlexNet_1D().cuda()
    model.load_state_dict(torch.load(PATH_Model))
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(1)
    model.to(device)

    

    X_test = torch.from_numpy(X).type(torch.FloatTensor)
    datatest = Variable(X_test.view(1,1,X_test.size(0))).to(device)


    outputs = model(datatest)
    _, predicted = torch.max(outputs.data, 1)

    result = Classes[int(predicted)]
    delta = int(result.replace("%",""))
    print('The error is: %s'%result)
    return delta

def Mono_10bit(vip,vin,Cp,Cn):
    code = []
    #print(code)
    Vref = 2
    Vcm = Vref/2
    Cp_sum = sum(Cp)
    Cn_sum = sum(Cn)
    for i in range(10):
    
        if(vip>vin):
            code.append("1")
            vip = vip - (Cp[i]/Cp_sum)*Vref
        else:
            code.append("0")
            vin = vin - (Cn[i]/Cn_sum)*Vref
    result = int("".join(code),2)
    #print(code)
    #print(result)
    return result
def Math_Function(delta):
    Ne = 16384*2
    #fi = int(re.findall(r'\d+', Path_Input.split("/")[-1])[0])*10**6
    fs = 1*10**9
    Me = 16383*0.4
    fi = fs*Me/Ne
    Ts = 1/fs
    Meg = 1
    delta = delta*0.01*Ts
    Code_fft = np.zeros(Ne, dtype=int)

    Cp = [256, 128, 64, 32, 16, 8, 4, 2, 1, 1] 
    Cn = [256, 128, 64, 32, 16, 8, 4, 2, 1, 1] 
    Cweight = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    for n in range(Ne):
        cal = Meg * math.sin(2*math.pi*fi*((n+1)*Ts + delta))
        vip = round((cal + 1),4)
        vin = round((-cal + 1),4) 
        Code_fft[n] = Mono_10bit(vip,vin,Cp,Cn)
        
    print(Code_fft)
if __name__ == "__main__":
    delta = Predict_TimeSkew()
    Math_Function(delta)
