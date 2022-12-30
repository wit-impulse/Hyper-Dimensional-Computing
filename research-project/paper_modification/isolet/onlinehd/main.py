import onlinehd
import pandas as pd
import sklearn.preprocessing
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import array
import pickle

# load the environment variables
epochs =50
lr = 0.035
dimension = 10000

with open('./dataset/isolet.pkl', 'rb') as f:
    isolet = pickle.load(f)
x, y, x_test, y_test = isolet

x = np.asarray(x)
y = np.asarray(y)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


# # load the dataset
# data = pd.read_csv('./dataset/data.csv', header=None)
# # retrieve data as numpy array
# values = data.values
# X = values[ :, :-1]
# y= values[:, -1]



# split and normalize
# x, x_test, y, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
scaler = sklearn.preprocessing.Normalizer().fit(x)
x = scaler.transform(x)
x_test = scaler.transform(x_test)


# changes data to pytorch's tensors
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).long()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()
print("Samples loaded successfully! Initiating the model training")



# load model 
classes = y.unique().size(0)
features = x.size(1)
model = onlinehd.OnlineHD(classes, features, dimension)

print('Training...')
model = model.fit(x, y, bootstrap=1.0, lr=lr, epochs=epochs)

print('Validating...')
yhat = model(x)
yhat_test = model(x_test)
acc = (y == yhat).float().mean()
acc_test = (y_test == yhat_test).float().mean()
print(f'{acc = :6f}')
print(f'{acc_test = :6f}')


