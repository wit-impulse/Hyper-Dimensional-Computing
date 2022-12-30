# importing the libraries
import numpy as np 


#PyTorch - Importing the Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms

# libraries for kpi
from ina260.controller import Controller
import sys
import os
import psutil


# mutiprocessing and time calculation libraries
from time import time
import multiprocess as mp
from multiprocessing import Process
import time as t

# result storage
import os.path
import csv


c = Controller(address=0x40)

def get_cpu_usage_pct():
    """
    Obtains the system's average CPU load as measured over a period of 500 milliseconds.
    :returns: System CPU load as a percentage.
    :rtype: float
    """
    return  float(psutil.cpu_percent(interval=0.5))

def get_cpu_frequency():
    """
    Obtains the real-time value of the current CPU frequency.
    :returns: Current CPU frequency in MHz.
    :rtype: int
    """
    stream = os.popen('vcgencmd measure_clock arm')
    output = stream.readlines()
    clock=output[0].split('=')
    #clock=commands.getoutput("vcgencmd measure_clock arm").split('=')
    clock=round((int(clock[1])/(10**6)),2) 
    return clock  
    
def get_cpu_temp():
    """
    Obtains the current value of the CPU temperature.
    :returns: Current value of the CPU temperature if successful, zero value otherwise.
    :rtype: float
    """
    #RPi command to get temperature
    stream = os.popen('vcgencmd measure_temp')
    output = stream.readlines()
    temp=output[0].split('=')  
    temp1=temp[1].split("'C")  
    #temp = commands.getoutput("vcgencmd measure_temp").split('=')
    temp=round((float(temp1[0])),2)
    return temp
    
def get_ram_total():
    """
    Obtains the total amount of RAM in bytes available to the system.
    :returns: Total system RAM usage as a percenatage.
    :rtype: int
    """
    return float(psutil.virtual_memory().percent)
def get_power():
    return round(c.power(),2)
    
def get_current():
    return round(c.current(),2)    

def get_voltage():
    return round(c.voltage(),2)    
def get_swap_usage_pct():
    """
    Obtains the system's current Swap usage.
    :returns: System Swap usage as a percentage.
    :rtype: float
    """
    return float(psutil.swap_memory().percent)







# functions to write KPI and test and train results in a csv file 
def performance_indicators(flag, model, mode, main_time):
    
    begin_time=time()
    while flag.value==0:
        temp=get_cpu_temp()
        cpu_freq=get_cpu_frequency()
        cpu_load= get_cpu_usage_pct()
        ram_usage= get_ram_total()
        swap_usage=get_swap_usage_pct()
        curr_time=time()-main_time
        curr_time=round(curr_time,2)
        volt=get_voltage()
        current=get_current()
        power=get_power()

        if os.path.isfile("results/indicators/"+mode+"/indicator.csv"):
            data=[model, mode, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage, volt, current, power]
            with open("results/indicators/"+mode+"/indicator.csv", 'a', newline='', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow(data)
                f.close()       
        else:
            print ("Creating result file...")
            header = ['model', 'mode','realtive time(sec)','temperature (Celcius)',
                      'cpu load(pct)','cpu freq(Hz)', 'ram usage(pct)','swap usage(pct)','voltage', 'current','power' ]
            
            data=[model, mode, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage, volt, current, power]
            with open("results/indicators/"+mode+"/indicator.csv", 'w', newline='', encoding='UTF8') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                #write the data
                writer.writerow(data)
                f.close()



def test_results(model_name, mode, acc_test, epochs, test_time, test_begin_time, test_end_time):
  if os.path.isfile("results/accuracy/test.csv"):
          #print("File exist writing data to existing file!")
          data=[model_name, mode, acc_test, epochs, test_time, test_begin_time, test_end_time]
          with open("results/accuracy/test.csv", 'a', newline='', encoding='UTF8') as f:
              writer = csv.writer(f)
              # write the data
              writer.writerow(data)
              f.close()
          
  else:
      print ("Creating result file...")
      header = ['model','mode','test accuracy','epochs','testing time','test begin time', 'test end time']  
      data=[model_name, mode, acc_test, epochs, test_time, test_begin_time, test_end_time]
      with open("results/accuracy/test.csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()






# load data
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset_pytorch = torchvision.datasets.FashionMNIST(root='./data/',
                                             train=True, 
                                             transform=transforms,
                                             download=True)
test_dataset_pytorch = torchvision.datasets.FashionMNIST(root='.data/',
                                             train=False, 
                                             transform=transforms,
                                             download=True)



train_loader = torch.utils.data.DataLoader(dataset=train_dataset_pytorch,
                                           batch_size=32, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset_pytorch,
                                           batch_size=32, 
                                           shuffle=False)


# build model
class NeuralNet(nn.Module):
    def __init__(self, num_of_class):
        super(NeuralNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5))
        self.fc_model = nn.Sequential(
            nn.Linear(726,20),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(20, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(-1, 1568)
        x = self.fc_model(x)
        x = self.classifier(x)
        return x


modelpy = NeuralNet(10)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(modelpy.parameters())



# variables
model_name= "cnn"
mode = "test"
epochs = 1


print("training...")
for e in range(epochs):
    # define the loss value after the epoch
    losss = 0.0
    number_of_sub_epoch = 0
    
    # loop for every training batch (one epoch)
    for images, labels in train_loader:
        #create the output from the network
        out = modelpy(images)
        # count the loss function
        loss = criterion(out, labels)
        # in pytorch you have assign the zero for gradien in any sub epoch
        optim.zero_grad()
        # count the backpropagation
        loss.backward()
        # learning
        optim.step()
        # add new value to the main loss
        losss += loss.item()
        number_of_sub_epoch += 1
    print("Epoch {}: Loss: {}".format(e, losss / number_of_sub_epoch))


print("temperature not in control....")
x= np.linspace(54.00, 55.00, 11)
while get_cpu_temp() not in x:
   t.sleep(1)
   
   
print("testing...")
main_time=time()
while time()-main_time<14400:
    
    flag=mp.Value("i", False)
    p1 = Process(target = performance_indicators, args=( flag, model_name, mode, main_time))
    p1.start()
    
    
    test_begin_time=time()-main_time
    test_time=time()
    # Testing the model
    correct = 0
    total = 0
    modelpy.eval()
    for images, labels in test_loader:
        outputs = modelpy(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc_test = 100 * correct / total
    
    test_time=time()-test_time
    test_end_time=time()-main_time
    with flag.get_lock():
        flag.value = True
    #store results
    print("Result stored")
    
    # round of to 2 digits
    acc_test= float(acc_test)
    acc_test=round(acc_test,2)
    test_time = round(test_time,2)
    test_begin_time= round(test_begin_time,2)
    test_end_time= round(test_end_time,2)
    
    test_results(model_name, mode, acc_test, epochs, test_time, test_begin_time, test_end_time)
    p1.join()
