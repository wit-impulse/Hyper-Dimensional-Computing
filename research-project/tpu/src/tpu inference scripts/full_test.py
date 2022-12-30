

import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import tensorflow as tf
import numpy as np
from ina260.controller import Controller
import os.path
import csv
from time import time as t
import sys
import os
import psutil
import multiprocess as mp
from multiprocessing import Process
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m', '--model', required=True, help='File path of .tflite file.')
parser.add_argument(
  '-l', '--labels', help='File path of labels file.')
parser.add_argument(
    '-k', '--top_k', type=int, default=1,
    help='Max number of classification results')
parser.add_argument(
    '-t', '--threshold', type=float, default=0.0,
    help='Classification score threshold')
parser.add_argument(
    '-c', '--count', type=int, default=1,
    help='Number of times to run inference')
parser.add_argument(
    '-a', '--input_mean', type=float, default=128.0,
    help='Mean value for input normalization')
parser.add_argument(
    '-s', '--input_std', type=float, default=128.0,
    help='STD value for input normalization')
args = parser.parse_args()

labels = read_label_file(args.labels) if args.labels else {}

interpreter = make_interpreter(*args.model.split('@'))
interpreter.allocate_tensors()

# Model must be uint8 quantized
if common.input_details(interpreter, 'dtype') != np.uint8:
  raise ValueError('Only support uint8 input type.')

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
images=test_images


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


#measure cpu temperature, frequency,...., total inference time, total accuracy
def performance_indicators(flag, samples, model, mode, main_time):

  while flag.value==0:
    temp=get_cpu_temp()
    cpu_freq=get_cpu_frequency()
    cpu_load= get_cpu_usage_pct()
    ram_usage= get_ram_total()
    swap_usage=get_swap_usage_pct()
    curr_time=t()-main_time
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
                  'cpu load(pct)','cpu freq(Hz)', 'ram usage(pct)','swap usage(pct)','voltage', 'current','power']
        
        data=[model, mode, samples, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage,volt, current, power]
        with open("results/indicators/"+mode+"/"+str(samples)+".csv", 'w', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            #write the data
            writer.writerow(data)
            f.close()

def test_results(model_name,mode,samples,accuracy,test_time, device):
    if os.path.isfile("results/accuracy/test.csv"):
          #print("File exist writing data to existing file!")
          data=[model_name,mode,samples,accuracy,test_time, device]
          with open("results/accuracy/test.csv", 'a', newline='', encoding='UTF8') as f:
              writer = csv.writer(f)
              # write the data
              writer.writerow(data)
              f.close()
    else:
      print ("Creating result file...")
      header = ['model','mode','samples','test accuracy','testing time', 'device']  
      data=[model_name,mode,samples,accuracy,test_time, device]
      with open("results/accuracy/test.csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()



tot_infer_time=0
pred=[]
accuracy=0


model_name='CNN'
mode='test'
device='edgetpu'
main_time=t()
samples=10000

x= np.linspace(55.00, 55.50, 6)
while get_cpu_temp() not in x:
   time.sleep(1)


flag=mp.Value("i", False)
p1 = Process(target = performance_indicators, args=( flag, samples, model_name, mode, main_time ))
p1.start() 


for image in images:

    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = args.input_mean
    std = args.input_std
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
        # Input data does not require preprocessing.
        common.set_input(interpreter, image)
    else:
        # Input data requires preprocessing
        normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
        np.clip(normalized_input, 0, 255, out=normalized_input)
        common.set_input(interpreter, normalized_input.astype(np.uint8))


    # Run inference
    #print('----INFERENCE TIME----')
    #print('Note: The first inference on Edge TPU is slow because it includes',
    #      'loading the model into Edge TPU memory.')
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        classes = classify.get_classes(interpreter, args.top_k, args.threshold)
        #print('%.1fms' % (inference_time * 1000))
        tot_infer_time=tot_infer_time+round((inference_time*1000),2)

#     print('-------RESULTS--------')
    for c in classes:
        #print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
        pred.append(labels.get(c.id, c.id))
    
with flag.get_lock():
    flag.value = True
p1.join()

tot_infer_time=round(tot_infer_time, 2)

pred=list(map(int,pred))
accuracy = (sum(1 for x,y in zip(test_labels,pred) if x == y  )* 100) / len(test_images)

test_results(model_name,mode,samples,accuracy,tot_infer_time, device)
# accuracy = (sum(1 for x,y in zip(test_labels,pred) if x == y  )* 100) / len(test_images)