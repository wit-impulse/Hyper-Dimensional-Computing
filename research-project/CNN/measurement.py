from ina260.controller import Controller
import psutil
import os
import os.path
import csv
from time import time

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
    clock=round((int(clock[1])/(10**6)),2) 
    return clock  
    
def get_cpu_temp():
    """
    Obtains the current value of the CPU temperature.
    :returns: Current value of the CPU temperature if successful, zero value otherwise.
    :rtype: float
    """
    stream = os.popen('vcgencmd measure_temp')
    output = stream.readlines()
    temp=output[0].split('=')  
    temp1=temp[1].split("'C")  
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
 
def performance_indicators(flag, samples, model, mode, main_time):
  begin_time=time()
  flag_freq=0
  while flag.value==0:
    temp= get_cpu_temp()
    cpu_freq= get_cpu_frequency()

    # if temp>60.0 and flag_freq==0:
    #     os.system('sudo cp /home/pi/Documents/minfreq /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq')
    #     flag_freq=1

    cpu_load= get_cpu_usage_pct()
    ram_usage= get_ram_total()
    swap_usage=get_swap_usage_pct()
    curr_time=time()-main_time
    curr_time=round(curr_time,2)
    volt=get_voltage()
    current=get_current()
    power=get_power()

    if os.path.isfile("results/indicators/"+mode+"/"+model+"/"+str(samples)+".csv"):
        data=[model, mode, samples, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage, volt, current, power]
        with open("results/indicators/"+mode+"/"+model+"/"+str(samples)+".csv", 'a', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the data
            writer.writerow(data)
            f.close()       
    else:
        print ("Creating result file...")
        header = ['model', 'mode', 'samples','realtive time(sec)','temperature (Celcius)',
                  'cpu load(pct)','cpu freq(Hz)', 'ram usage(pct)','swap usage(pct)','voltage', 'current','power' ]
        
        data=[model, mode, samples, curr_time, temp, cpu_load, cpu_freq, ram_usage, swap_usage,volt, current, power]
        with open("results/indicators/"+mode+"/"+model+"/"+str(samples)+".csv", 'w', newline='', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            #write the data
            writer.writerow(data)
            f.close()


def test_results(model_name, mode, samples, acc_test, epochs, dimension, lr, device, test_time):
  if os.path.isfile("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv"):
          #print("File exist writing data to existing file!")
          data=[model_name, samples, acc_test, epochs, dimension, lr, device, test_time]
          with open("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv", 'a', newline='', encoding='UTF8') as f:
              writer = csv.writer(f)
              # write the data
              writer.writerow(data)
              f.close()
          
  else:
      print ("Creating result file...")
      header = ['model','samples','test accuracy','epochs',
                'dimension','learning rate', 'device','testing time']  
      data=[model_name, samples, acc_test, epochs, dimension, lr, device, test_time]
      with open("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()


def train_results(model_name, mode, samples, main_time, epochs, dimension, lr, device, train_time, dl_begin_time,dl_end_time,tb_begin_time,tb_end_time):
  if os.path.isfile("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv"):
          #print("File exist writing data to existing file!")
          data=[model_name, samples,epochs, dimension, lr, device, train_time,dl_begin_time,dl_end_time,tb_begin_time,tb_end_time]
          with open("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv", 'a', newline='', encoding='UTF8') as f:
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
      with open("results/accuracy/"+mode+"/"+model_name+"/"+str(samples)+".csv", 'w', newline='', encoding='UTF8') as f:
          writer = csv.writer(f)
          # write the header
          writer.writerow(header)
          #write the data
          writer.writerow(data)
          f.close()