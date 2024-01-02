import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import random as rn
import math
from itertools import combinations
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss
# From KDD2020 calibration tutorial
from scipy.special import softmax
from sklearn.metrics import log_loss
from medmnist import INFO, Evaluator
import dataset_without_pytorch

from scipy.optimize import minimize

import matplotlib.pyplot as plt

### same as ood
def init_random_seeds(no_cuda):
  os.environ['PYTHONHASHSEED'] = '0'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  if no_cuda: # supposedly this use of GPU will make random seed useless
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  np.random.seed(37)
  rn.seed(1254)
  tf.random.set_seed(89)

### same as ood
def expand(X_train):
  #Convert the data to a format compatible with Keras (which expects additional dim)
  X_train_exp = np.expand_dims(X_train, -1)
  return(X_train_exp)

### same as ood
def med_one_hot(X_train, Y_train, num_classes):
  X_train_exp = np.expand_dims(X_train, -1)
  Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)
  return(X_train_exp, Y_train_cat)

### same as ood
# The CNN
initial_learning_rate = 0.01
def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)

### same as ood
def setup_model(img_size, num_classes):
  model = tf.keras.Sequential(
     [
       tf.keras.Input(shape=img_size),
       tf.keras.layers.Conv2D(40, kernel_size = (3,3), activation = "relu"),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
       tf.keras.layers.Conv2D(80, kernel_size = (3,3), activation = "relu"),
       tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(num_classes, activation="softmax")     
     ]
  )
  return model

def make_test_ood_classes(id_classes, train_ood_classes):
  if not isinstance(id_classes[0], int):
    print('ood_use_classes_balanced expects numeric list of classes')
    sys.exit()

  test_ood_classes = list(np.arange(9))
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('make_test_ood_classes:', 'num_id_classes', len(id_classes), 'num_ood_classes', len(train_ood_classes))
  print('make_test_ood_classes:', id_classes, train_ood_classes, test_ood_classes)
  return test_ood_classes

def med_use_classes_control(id_classes, train_dataset):
  x_train, y_train = med_get_raw_data_as_classes(train_dataset)
  x = x_train[id_classes[0]]
  y = y_train[id_classes[0]]
  for i in range(1,len(id_classes)):
    x = np.vstack((x, x_train[id_classes[i]]))
    y = np.hstack((y, y_train[id_classes[i]]))

  print('med_use_classes_control:', len(x), len(y))

  z = list(zip(x,y))
  np.random.shuffle(z)
  x, y = zip(*z)
  x = np.asarray(x)
  y = np.asarray(y)

  print('med_use_classes_control:', len(x), len(y), x.shape, y.shape)

  return x, y

def med_use_classes_balanced(id_classes, train_ood_classes, train_dataset):
  x_train, y_train = med_get_raw_data_as_classes(train_dataset)
  x_id = x_train[id_classes[0]]
  y_id = y_train[id_classes[0]]
  for i in range(1,len(id_classes)):
    x_id = np.vstack((x_id, x_train[id_classes[i]]))
    y_id = np.hstack((y_id, y_train[id_classes[i]]))
  x_train_ood = x_train[train_ood_classes[0]]
  y_train_ood = y_train[train_ood_classes[0]]
  for i in range(1,len(train_ood_classes)):
    x_train_ood = np.vstack((x_train_ood, x_train[train_ood_classes[i]]))
    y_train_ood = np.hstack((y_train_ood, y_train[train_ood_classes[i]]))

  train_ood_sub = list(zip(x_train_ood, y_train_ood))
  np.random.shuffle(train_ood_sub)
  x_train_ood, y_train_ood = zip(*train_ood_sub)
  sel_num = len(x_id)
  sel_num = int(sel_num/len(id_classes))
  x_train_ood = x_train_ood[0:sel_num]
  y_train_ood = y_train_ood[0:sel_num]

  print('med_use_classes_balanced:', len(x_id), len(y_id), len(x_train_ood), len(y_train_ood))

  x = np.vstack((x_id, x_train_ood))
  y = np.hstack((y_id, y_train_ood))

  z = list(zip(x,y))
  np.random.shuffle(z)
  x, y = zip(*z)
  x = np.asarray(x)
  y = np.asarray(y)

  print('med_use_classes_balanced:', len(x), len(y), x.shape, y.shape)

  return x, y

def med_use_classes_unbalanced(id_classes, train_ood_classes, train_dataset):
  x_train, y_train = med_get_raw_data_as_classes(train_dataset)
  x_id = x_train[id_classes[0]]
  y_id = y_train[id_classes[0]]
  for i in range(1,len(id_classes)):
    x_id = np.vstack((x_id, x_train[id_classes[i]]))
    y_id = np.hstack((y_id, y_train[id_classes[i]]))
  x_train_ood = x_train[train_ood_classes[0]]
  y_train_ood = y_train[train_ood_classes[0]]
  for i in range(1,len(train_ood_classes)):
    x_train_ood = np.vstack((x_train_ood, x_train[train_ood_classes[i]]))
    y_train_ood = np.hstack((y_train_ood, y_train[train_ood_classes[i]]))

  print('med_use_classes_unbalanced:', len(x_id), len(y_id), len(x_train_ood), len(y_train_ood))

  x = np.vstack((x_id, x_train_ood))
  y = np.hstack((y_id, y_train_ood))

  z = list(zip(x,y))
  np.random.shuffle(z)
  x, y = zip(*z)
  x = np.asarray(x)
  y = np.asarray(y)

  print('med_use_classes_unbalanced:', len(x), len(y), x.shape, y.shape)

  return x, y

def med_ood_make_cases234(char_classes_map):
  all_classes = list(char_classes_map.values())
  id_comb = combinations(all_classes, 2)
  cases = []
  for id_cl in id_comb:
    this_classes = all_classes.copy()
    this_classes.remove(id_cl[0])
    this_classes.remove(id_cl[1])
    train_ood_comb = combinations(this_classes, 3)
    for train_ood_cl in train_ood_comb:
      cases.append((id_cl,train_ood_cl))
  return cases

def med_get_datasets():
  data_flag = 'pathmnist' # 'chestmnist' # 
  download = True
  info = INFO[data_flag]
  DataClass = getattr(dataset_without_pytorch, info['python_class'])
  # load the data
  train_dataset = DataClass(split='train', download=download)
  val_dataset = DataClass(split='val', download=download)
  test_dataset = DataClass(split='test', download=download)
  return train_dataset, val_dataset, test_dataset

def med_get_raw_data_from_dataset(dataset):
  x = []
  y = []
  for i in range(len(dataset)):
    dx, dy = dataset[i]
    x.append(np.asarray(dx))
    y.append(int(dy))
    
  x = np.asarray(x)
  x = x.astype("float32") / 255

  y = np.asarray(y, dtype=np.uint8)

  return x, y

def med_split_data(x, y):
  uniq = np.unique(y)
  assert len(uniq) == 9
  retx = [0]*len(uniq)
  rety = [0]*len(uniq)
  for cl in uniq:
    mask = np.isin(y, cl)
    x_cl, y_cl = x[mask], y[mask]
    retx[cl] = x_cl
    rety[cl] = y_cl
  return retx, rety

def med_get_raw_data_as_classes(dataset):
  x, y = med_get_raw_data_from_dataset(dataset)
  x, y = med_split_data(x, y)
  return x, y

def med_train(model, X_train, Y_train):
  batch_size = 128
  epochs = 60 # 15
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  hist = model.fit(X_train, Y_train, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1)# , # callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return hist

# note that Y_train is the 0, 1, 2 version of the y_train
def calc_threshold(X_train, Y_train, model, num_classes, type):

  thres_vec = []
  thres_max = 0
  thres_min = 1
  X_train_exp = expand(X_train)
  y_pred_train = model.predict(X_train_exp)
  for i in range(num_classes):
    y_pred_train_i = np.isin(Y_train, i)
    thres_i = np.mean(y_pred_train[y_pred_train_i][:,i])
    if thres_i > thres_max: thres_max = thres_i
    if thres_i < thres_min: thres_min = thres_i
    thres_vec.append(thres_i)
  thres = np.mean(thres_vec)
  print('mean', thres, 'max', thres_max, 'min', thres_min)
  if type == 'mean':
    return thres
  elif type == 'max':
    return thres_max
  elif type == 'min':
    return thres_min
  elif type == 'ood':
    return thres_vec[2]
  else:
    print('error')
    sys.exit()

def med_print_count(ind, y):
  nums = [0]*9
  for i in range(len(ind)):
    nums[(int(y[ind[i]]))] += 1
  print(nums)

def med_print_count_noind(y):
  nums = [0]*9
  for i in range(len(y)):
    nums[int(y[i])] += 1
  print(nums)

def med_find_new_class(input_index, y_pred, c, max_thres, min_thres):
  i=0
  index=[]
  #get all training imgs and add the twos from the test imgs to it
  while i<len(y_pred):
    if i in input_index:
      if y_pred[i,c] < max_thres and y_pred[i,c] > min_thres:
        index.append(i)
    i=i+1
  return index

def medd_make_new_class(index, x_test, X_train, Y_train, c):
  for i in index:
    #X_test1 = np.expand_dims(x_test[i], axis=0)
    p=x_test[i]
    X = p[np.newaxis, :, :]
    print(X_train.shape, X.shape)
    X_train = np.vstack((X_train, X))
    Y_train = np.append(Y_train, [c])
  return X_train, Y_train

def med_add_new_class(ind, x_test, Y_test, X_train, Y_train, c):
  i=0
  j=0
  for i in ind:
    p=x_test[i]
    X = p[np.newaxis, :, :]
    #print(X_train.shape, X.shape)
    X_train = np.vstack((X_train, X))
    Y_train = np.append(Y_train, [c])
    i=i+1
  return(X_train, Y_train)

def mod_calc_perc(num_classes, id_classes, train_ood_classes, test_ood_classes, y_pred, y_test):
  assert(num_classes == 2)
  id = np.isin(y_test, id_classes)
  train_ood = np.isin(y_test, train_ood_classes)
  test_ood = np.isin(y_test, test_ood_classes)
  x = y_pred[id]
  print(x.sum())
  x = y_pred[train_ood]
  print(x.sum())
  x = y_pred[test_ood]
  print(x.sum())
  #print(len(y_test[id]), len(y_test[train_ood]), len(y_test[test_ood]))

  

def calc_percentages(num_classes, id_classes, train_ood_classes, test_ood_classes, y_pred, y_test):
  id_val = 0
  id_count = 0
  train_ood_val = 0
  train_ood_count = 0
  test_ood_val = 0
  test_ood_count = 0
  res = []
  for cl in range(9): #9 for med, 10 for mnist
    corr = 0
    cs = 0
    for i in range(len(y_pred)):
      if y_test[i] == cl:
        cs = cs + 1
        if y_pred[i] == len(id_classes):
          corr = corr + 1
    mask = np.isin(y_test, cl)
    masked_y_pred = y_pred[mask]
    nums = [0]*num_classes
    for i in range(len(masked_y_pred)):
      nums[masked_y_pred[i]] += 1

    if cl in id_classes:
      id_count = id_count + 1
      id_val = id_val + corr / cs
      res.append(('  id', cl, corr/cs, nums))
    if cl in train_ood_classes:
      train_ood_count = train_ood_count + 1
      train_ood_val = train_ood_val + corr / cs
      res.append((' ood', cl, corr/cs, nums))
    if cl in test_ood_classes:
      test_ood_count = test_ood_count + 1
      test_ood_val = test_ood_val + corr / cs
      res.append(('test', cl, corr/cs, nums))

  return id_val, id_count, train_ood_val, train_ood_count, test_ood_val, test_ood_count, res


def med_pred_percentage(y, pred, new, ood):
  corr=0
  cs=0
  for i in range(len(pred)):
    if y[i]==new:
      cs=cs+1
      if pred[i]==ood:
        corr=corr+1
  print(new, corr/cs)
  return(cs, corr)

