#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
import multiprocess as mp
from multiprocessing import Process
import tensorflow as tf
import numpy as np
import time as t
import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
import tensorflow as tf
import numpy as np
import torch
from collections import Counter
from scipy.io import loadmat
import os.path
import csv
from time import time
import sys
import os
import psutil
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from ina260.controller import Controller


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
#balanced dataset sampler    

def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train=X[balanced_copy_idx]
    labels_train=y[balanced_copy_idx]
    if  ((len(data_train)) == (sample_size*len(uniq_levels))):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size, ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    check = all(x == values[0] for x in values)
    if check == True:
        print('All classes have same samples:check complete')
    else:
        print('Repeat again your sampling your classes are not balanced')
    return data_train,labels_train    

#fashion mnist dataset
def fmnist(samples):
    # fetches data
    (x, y), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #x, y = sklearn.datasets.fetch_openml('mnist_784', return_X_y=True)
    
    #reduce tarining data according to samples required
    #x,y=balanced_sample_maker(x,y,samples)
    
    x = x.astype(np.float)/255.0
    x_test = x_test.astype(np.float)/255.0       
    y = y.astype(np.int)
    y_test = y_test.astype(np.int)
          
    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x = x.reshape(x.shape[0], w, h, 1)
    #x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)
    
    # One-hot encode the labels
    y = tf.keras.utils.to_categorical(y, 10)
    #y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x, x_test, y, y_test


# loads SVHN dataset
def svhn(samples):
    # fetches data
    train_raw = loadmat('data/train_32x32.mat')
    test_raw = loadmat('data/test_32x32.mat')
    
    x = np.array(train_raw['X'])
    x_test = np.array(test_raw['X'])
    
    y = train_raw['y']
    y_test= test_raw['y']
    
    x = np.moveaxis(x, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
     
    y=y.reshape(1,y.shape[0])[0]
    y_test=y_test.reshape(1,y_test.shape[0])[0]
    
    x,y=balanced_sample_maker(x,y,samples)
    
    x = x.astype('float64')
    x_test = x_test.astype('float64')
    # Convert train and test labels into 'int64' type    
    y = y.astype('int64')
    y_test = y_test.astype('int64')
    x /= 255.0
    x_test /= 255.0
    # One-hot encoding of train and test labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    y_test = lb.fit_transform(y_test)
    return x, x_test, y, y_test


def load_dataset(dataset, samples):
  if dataset=='fmnist':
    x, x_test, y, y_test= fmnist(samples)

  elif dataset=='svhn':
    x, x_test, y, y_test= svhn(samples)

  else:
     sys.exit("dataset not found")
  return x, x_test, y, y_test


def performance_indicators(flag, samples, model, mode, main_time):
  begin_time=time()
  flag_freq=0
  while flag.value==0:
    temp=get_cpu_temp()
    cpu_freq=get_cpu_frequency()
    if temp>60.0 and flag_freq==0:
        os.system('sudo cp /home/pi/Documents/maxfreq /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')
        flag_freq=1
    cpu_load= get_cpu_usage_pct()
    ram_usage= get_ram_total()
    swap_usage=get_swap_usage_pct()
    curr_time=time()-main_time
    curr_time=round(curr_time,2)
    volt=get_voltage()
    current=get_current()
    power=get_power()

    if os.path.isfile("results/indicators/"+mode+"/"+str(samples)+".csv"):
        data=[model, mode, samples, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage, volt, current, power]
        with open("results/indicators/"+mode+"/"+str(samples)+".csv", 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the data
            writer.writerow(data)
            f.close()       
    else:
        print ("Creating result file...")
        header = ['model', 'mode', 'samples','realtive time(sec)','temperature (Celcius)',
                  'cpu load(pct)','cpu freq(Hz)', 'ram usage(pct)','swap usage(pct)','voltage', 'current','power' ]
        
        data=[model, mode, samples, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage,volt, current, power]
        with open("results/indicators/"+mode+"/"+str(samples)+".csv", 'w', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            #write the data
            writer.writerow(data)
            f.close()


def test_results(model_name, mode, samples, acc_test, epochs, dimension, lr, device, test_time):
  if os.path.isfile("results/accuracy/test.csv"):
          #print("File exist writing data to existing file!")
          data=[model_name, samples, acc_test, epochs, dimension, lr, device, test_time]
          with open("results/accuracy/test.csv", 'a', newline='', encoding='UTF8') as f:
              writer = csv.writer(f)
              # write the data
              writer.writerow(data)
              f.close()
          
  else:
      print ("Creating result file...")
      header = ['model','samples','test accuracy','epochs',
                'dimension','learning rate', 'device','testing time']  
      data=[model_name, samples, acc_test, epochs, dimension, lr, device, test_time]
      with open("results/accuracy/test.csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()


def train_results(model_name, mode, samples, main_time, epochs, dimension, lr, device, train_time, dl_begin_time,dl_end_time,tb_begin_time,tb_end_time):
  if os.path.isfile("results/accuracy/train.csv"):
          #print("File exist writing data to existing file!")
          data=[model_name, samples,epochs, dimension, lr, device, train_time,dl_begin_time,dl_end_time,tb_begin_time,tb_end_time]
          with open("results/accuracy/train.csv", 'a', newline='', encoding='UTF8') as f:
              writer = csv.writer(f)
              # write the data
              writer.writerow(data)
              f.close()
          
  else:
      print ("Creating result file...")
      header = ['model','samples','epochs',
                'dimension','learning rate', 'device','training time',
                'dl begin','dl end', 'train begin', 'train end']  
      data=[model_name, samples, epochs, dimension, lr,
            device, train_time, dl_begin_time,
            dl_end_time,tb_begin_time,tb_end_time]
      with open("results/accuracy/train.csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()


# In[2]:


# x= np.linspace(54.00, 55.00, 11)
# 
# while get_cpu_temp() not in x:
#   t.sleep(1)
# print("CPU Temperature under control...")

"""
enter the required variables
"""
samples=6000
model_name='cnn'    
dataset='fmnist'
epochs=1
lr=0.001
dimension=0
device='raspberrypi'
mode='train'

batch_size=64


i=0
z= np.linspace(54.00, 55.00, 11)
while get_cpu_temp() not in z:
    t.sleep(1)
    
main_time=time()
while time()-main_time<14400:
    flag=mp.Value("i", False)
    p1 = Process(target = performance_indicators, args=( flag, samples, model_name, mode, main_time ))
    p1.start()
    #dl_main_time=main_time()
    dl_begin_time=time()-main_time
    dl_begin_time=round(dl_begin_time, 2)
    


    if mode=='train':
          #train_time
          
          #p1.start()
          x,x_test,y,y_test = load_dataset(dataset, samples)
          print("Samples loading complete!")
          
          dl_end_time=time()-main_time
          #tb_begin_time=time()-dl_main_time
          dl_end_time=round(dl_end_time, 2)
          #tb_begin_time=round(tb_begin_time, 2)
          #size=x.shape
    
    """
    Initialize the 2 processes which share the common flag. P2 will stop measuring performance indicators when the
    flag is False which is done by the first process.
    """ 

    if mode=='train':
      #train_time

      #x,x_test,y,y_test = load_dataset(dataset, samples)
      #print("Samples loading complete!")
      
      #dl_end_time=time()-main_time
      tb_begin_time=time()-main_time
      #dl_end_time=round(dl_end_time, 2)
      tb_begin_time=round(tb_begin_time, 2)
      size=x.shape

      model = tf.keras.Sequential()
      model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(size[1], size[2], size[3])))
                
      model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
      model.add(tf.keras.layers.Dropout(0.3))
      
      model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
      model.add(tf.keras.layers.Dropout(0.3))
      
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(256, activation='relu'))
      model.add(tf.keras.layers.Dropout(0.5))
      model.add(tf.keras.layers.Dense(10, activation='softmax'))


      optimizer = keras.optimizers.Adam(learning_rate=lr)
      model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
      train_time = time()
      history= model.fit(x, y, batch_size= batch_size, epochs=epochs)
      train_time = time()-train_time
      
      tb_end_time= time()-main_time
      tb_end_time=round(tb_end_time, 2)
      with flag.get_lock():
        flag.value = True

      

#       train_eval_time=time()
#       score = model.evaluate(x, y, verbose=0)   
#       train_eval_time=time()-train_eval_time

      #round off the numbers
#       acc=float(score[1])
#       acc=round(acc*100,2)
      train_time=round(train_time, 2)
#       train_eval_time=round(train_eval_time,2)

      train_results(model_name, mode, samples, main_time, epochs, dimension, lr, device, train_time, dl_begin_time,dl_end_time,tb_begin_time,tb_end_time)
      print("results are stored!")
      p1.join()
    

# In[26]:


# x= np.linspace(54.00, 55.00, 11)
# 
# while get_cpu_temp() not in x:
#   t.sleep(1)
# 
# print("CPU Temperature under control...")
# flag=mp.Value("i", False)
# mode='test'
# if mode=='test':
# 
#   print('Validating...')
#   p1 = Process(target = performance_indicators, args=( flag, samples, model_name, mode))
#   p1.start()
#   
#   t.sleep(3)
# 
#   test_time=time()
#   score = model.evaluate(x_test, y_test, verbose=0)
#   y_pred = model.predict(x_test)
#   test_time=time()-test_time
# 
#   with flag.get_lock():
#     flag.value = True
# 
#   p1.join()
# 
#   
#   #round off the numbers
#   acc_test=float(score[1])
#   acc_test=round(acc_test*100,2)
#   test_time=round(test_time,2)
# 
#   test_results(model_name, mode, samples, acc_test, epochs, dimension, lr, device, test_time)
#   print("results are stored!")

