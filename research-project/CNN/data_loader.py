from mnist import MNIST
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
from scipy.io import loadmat


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
    #(x, y), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    mndata = MNIST('../data')
    x, y = mndata.load_training()
    x_test, y_test = mndata.load_testing()

    x=np.array(x)
    x_test=np.array(x_test)

    x=x.reshape(60000,28,28)
    x_test=x_test.reshape(10000,28,28)

    #reduce tarining data according to samples required
    if samples !=6000:
        x,y=balanced_sample_maker(x,y,samples)
    
    x = x.astype(np.float)/255.0
    x_test = x_test.astype(np.float)/255.0       
    #y = y.astype(np.int)
    #y_test = y_test.astype(np.int)
          
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