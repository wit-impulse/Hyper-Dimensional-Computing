import onlinehd
import numpy as np
from time import time
import time as t
from os import environ as env
from dotenv import load_dotenv
from measurement import get_cpu_temp, performance_indicators, test_results, train_results
from data_loader import load_dataset
import multiprocess as mp
from multiprocessing import Process

# load the environment variables
load_dotenv()
samples = int(env['TRAINING_SAMPLES'])
model_name = env['MODEL']
dataset = env['DATASET']
epochs = int(env['EPOCHS'])
lr = float(env['LEARNING_RATE'])
mode = env['MODE']
dimension = int(env['DIMENSION'])
# batch_size = int(env['BATCH_SIZE'])
duration = int(env['DURATION'])
device="pi"

# make sure that code starts between 54 and 55 degree celcius
print("waiting for CPU temperature to come in range 54-55")
z = np.linspace(54.00, 55.00, 11)
while get_cpu_temp() not in z:
    t.sleep(1)


if mode == "train":
    iteration=0

    # data loading first time
    main_time = time()
    dl_begin_time = time()-main_time
    dl_begin_time = round(dl_begin_time, 2)
    x, x_test, y, y_test= load_dataset(dataset, samples)
    #size=x.shape
    print("Samples loaded successfully! Initiating the model training")
    print("Training...")
    dl_end_time=time()-main_time
    dl_end_time=round(dl_end_time, 2)

    main_time = time()
    while time()-main_time < duration:
        # start the multiprocess to measure the KPIs
        flag = mp.Value("i", False)
        p1 = Process(target=performance_indicators, args=(
            flag, samples, model_name, mode, main_time))
        p1.start()

        #training
        tb_begin_time=time()-main_time
        tb_begin_time=round(tb_begin_time, 2)
        #load model 
        classes = y.unique().size(0)
        features = x.size(1)
        model = onlinehd.OnlineHD(classes, features, dimension)
        train_time = time()
        model = model.fit(x, y, bootstrap=1.0, lr=lr, epochs=epochs)
        train_time = time()-train_time
        tb_end_time= time()-main_time
        tb_end_time=round(tb_end_time, 2)
        train_time=round(train_time, 2)
        
        train_results(model_name, mode, samples, main_time, epochs, dimension, lr, device, train_time, dl_begin_time, dl_end_time, tb_begin_time, tb_end_time)
        print("iteration "+str(iteration)+" results stored! Starting next iteration")
        iteration=iteration+1
        with flag.get_lock():
            flag.value = True
        p1.join()

elif mode == "test":

    iteration=0
    # data loading first time
    main_time = time()
    dl_begin_time = time()-main_time
    dl_begin_time = round(dl_begin_time, 2)
    x, x_test, y, y_test= load_dataset(dataset, samples)
    #size=x.shape
    print("Samples loaded successfully! Initiating the model training")
    print("Training...")
    dl_end_time=time()-main_time
    dl_end_time=round(dl_end_time, 2)


    #training
    tb_begin_time=time()-main_time
    tb_begin_time=round(tb_begin_time, 2)
    #load model 
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(classes, features, dimension)
    
    train_time = time()
    model = model.fit(x, y, bootstrap=1.0, lr=lr, epochs=epochs)
    train_time = time()-train_time
    tb_end_time= time()-main_time
    tb_end_time=round(tb_end_time, 2)
    train_time=round(train_time, 2)


    main_time = time()
    while time()-main_time < duration:
        # start the multiprocess to measure the KPIs
        flag = mp.Value("i", False)
        p1 = Process(target=performance_indicators, args=(
            flag, samples, model_name, mode, main_time))
        p1.start()

        #testing
        test_begin_time=time()-main_time
        test_time=time()
        yhat_test = model(x_test)
        acc_test = (y_test == yhat_test).float().mean()
        test_time=time()-test_time
        test_end_time=time()-main_time

        #round off the numbers
        acc_test=float(acc_test)
        acc_test=round(acc_test*100,2)
        test_time=round(test_time,2)
        test_begin_time=round(test_begin_time,2)
        test_end_time=round(test_end_time,2)

        test_results(model_name, mode, samples, acc_test, epochs, dimension, lr, device, test_time)
        print("iteration "+str(iteration)+" results stored! Starting next iteration")
        iteration=iteration+1
        with flag.get_lock():
            flag.value = True
        p1.join()

        
else:
    iteration=0
    main_time = time()
    while time()-main_time < duration:
        # start the multiprocess to measure the KPIs
        flag = mp.Value("i", False)
        p1 = Process(target=performance_indicators, args=(
            flag, samples, model_name, mode, main_time))
        p1.start()

        # data loading
        dl_begin_time = time()-main_time
        dl_begin_time = round(dl_begin_time, 2)
        x, x_test, y, y_test= load_dataset(dataset, samples)
        #size=x.shape
        print("Samples loaded successfully! Initiating the model training")
        print("Training...")
        dl_end_time=time()-main_time
        dl_end_time=round(dl_end_time, 2)


        #training
        tb_begin_time=time()-main_time
        tb_begin_time=round(tb_begin_time, 2)
        #load model 
        classes = y.unique().size(0)
        features = x.size(1)
        model = onlinehd.OnlineHD(classes, features, dimension)

        train_time = time()
        model = model.fit(x, y, bootstrap=1.0, lr=lr, epochs=epochs)
        train_time = time()-train_time
        tb_end_time= time()-main_time
        tb_end_time=round(tb_end_time, 2)
        train_time=round(train_time, 2)
        
        train_results(model_name, mode, samples, main_time, epochs, dimension, lr, device, train_time, dl_begin_time, dl_end_time, tb_begin_time, tb_end_time)
        print("iteration "+str(iteration)+" results stored! Starting next iteration")
        iteration=iteration+1
        with flag.get_lock():
            flag.value = True
        p1.join()
    
