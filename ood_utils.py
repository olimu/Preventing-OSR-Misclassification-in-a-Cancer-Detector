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

from scipy.optimize import minimize

import matplotlib.pyplot as plt

def init_random_seeds(no_cuda):
  os.environ['PYTHONHASHSEED'] = '0'
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  if no_cuda: # supposedly this use of GPU will make random seed useless
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  np.random.seed(37)
  rn.seed(1254)
  tf.random.set_seed(89)

def make_test_ood_classes(id_classes, train_ood_classes):
  if not isinstance(id_classes[0], int):
    print('ood_use_classes_balanced expects numeric list of classes')
    sys.exit()

  test_ood_classes = list(np.arange(10))
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', len(id_classes), 'num_ood_classes', len(train_ood_classes))
  print(id_classes, train_ood_classes, test_ood_classes)
  return test_ood_classes

def ood_use_classes_control(id_classes, train_ood_classes):

  if not isinstance(id_classes[0], int):
    print('ood_use_classes_balanced expects numeric list of classes')
    sys.exit()

  test_ood_classes = list(np.arange(10))
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', len(id_classes), 'num_ood_classes', len(train_ood_classes))
  print(id_classes, train_ood_classes, test_ood_classes)

  x_train, y_train, x_test, y_test = get_raw_data_as_classes()
  x = x_train[id_classes[0]]
  y = y_train[id_classes[0]]
  for i in range(1,len(id_classes)):
    x = np.vstack((x, x_train[id_classes[i]]))
    y = np.hstack((y, y_train[id_classes[i]]))

  print('ood_use_classes_control', len(x), len(y))

  z = list(zip(x,y))
  np.random.shuffle(z)
  x, y = zip(*z)
  x = np.asarray(x)
  y = np.asarray(y)

  print('ood_use_classes_control', len(x), len(y), x.shape, y.shape)

  return x, y, test_ood_classes

def ood_use_classes_balanced(id_classes, train_ood_classes):

  if not isinstance(id_classes[0], int):
    print('ood_use_classes_balanced expects numeric list of classes')
    sys.exit()

  test_ood_classes = list(np.arange(10))
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', len(id_classes), 'num_ood_classes', len(train_ood_classes))
  print(id_classes, train_ood_classes, test_ood_classes)

  x_train, y_train, x_test, y_test = get_raw_data_as_classes()
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

  print(len(x_id), len(y_id), len(x_train_ood), len(y_train_ood))

  x = np.vstack((x_id, x_train_ood))
  y = np.hstack((y_id, y_train_ood))

  z = list(zip(x,y))
  np.random.shuffle(z)
  x, y = zip(*z)
  x = np.asarray(x)
  y = np.asarray(y)

  print(len(x), len(y), x.shape, y.shape)

  return x, y, test_ood_classes


def ood_use_classes_orig(id_classes, train_ood_classes, char_classes_map):

  test_ood_classes = list(char_classes_map.values())
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', len(id_classes), 'num_ood_classes', len(train_ood_classes))
  print(id_classes, train_ood_classes, test_ood_classes)

  id_class_labels = [x for x in range(len(id_classes))]
  train_ood_class_labels = np.ones(len(train_ood_classes))*len(id_classes)
  test_ood_class_labels = np.ones(len(test_ood_classes))*len(id_classes)

  print(id_class_labels, train_ood_class_labels, test_ood_class_labels)

  keys = np.concatenate((id_classes, train_ood_classes, test_ood_classes))
  print('keys', keys)
  values = np.concatenate((id_class_labels, train_ood_class_labels, test_ood_class_labels))
  print('values', values)

  assert len(keys) == len(values)
  num_classes_map = {}
  for i in range(len(keys)):
    num_classes_map[keys[i]] = int(values[i])
  print(num_classes_map)

  legal_classes = id_classes + train_ood_classes

  # get the raw mnist data and label classes as alphabetical characters
  x_train, y_train, x_test, y_test = get_raw_data()
  y_train_char = [char_classes_map[e] for e in y_train]
  y_test_char = [char_classes_map[e] for e in y_test]

  train_mask = np.isin(y_train_char, legal_classes)
  y_train_char = np.asarray(y_train_char)
  y_train_char_subset = y_train_char[train_mask]
  Y_train = [num_classes_map[e] for e in y_train_char_subset]
  X_train = x_train[train_mask]
  print(len(X_train), len(Y_train), X_train.shape)
  #Y_test = [num_classes_map[e] for e in y_train_char]
  num_classes = len(id_classes) + 1
  X_train_exp, Y_train_cat = one_hot(X_train, Y_train, num_classes-1) # one_hot creates one more class
  img_size = X_train_exp[0].shape

  return num_classes_map, test_ood_classes, img_size, num_classes, x_test, y_test, X_train_exp, Y_train_cat

def ood_make_classes_general():
  # the MNIST numbers set has 10 classes, we use characters to temporarily rename them
  char_classes_map = { 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j' }
  num_id_classes = rn.randint(1, 3)
  num_ood_classes = rn.randint(num_id_classes, 7)
  if num_id_classes + num_ood_classes < 10:
    pass
  else:
    num_ood_classes = min(4, num_ood_classes) 

  seq = list(range(10))
  id_classes = []
  for i in range(num_id_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    id_classes.append(char_classes_map[choice])
  train_ood_classes = []
  for i in range(num_ood_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    train_ood_classes.append(char_classes_map[choice])
  test_ood_classes = list(char_classes_map.values())
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', num_id_classes, 'num_ood_classes', num_ood_classes)
  print(id_classes, train_ood_classes, test_ood_classes)

  id_class_labels = [x for x in range(len(id_classes))]
  train_ood_class_labels = np.ones(len(train_ood_classes))*len(id_classes)
  test_ood_class_labels = np.ones(len(test_ood_classes))*len(id_classes)

  print(id_class_labels, train_ood_class_labels, test_ood_class_labels)

  keys = np.concatenate((id_classes, train_ood_classes, test_ood_classes))
  print('keys', keys)
  values = np.concatenate((id_class_labels, train_ood_class_labels, test_ood_class_labels))
  print('values', values)
  assert len(keys) == len(values)
  num_classes_map = {}
  for i in range(len(keys)):
    num_classes_map[keys[i]] = int(values[i])
  print(num_classes_map)

  legal_classes = id_classes + train_ood_classes

  # get the raw mnist data and label classes as alphabetical characters
  x_train, y_train, x_test, y_test = get_raw_data()
  y_train_char = [char_classes_map[e] for e in y_train]
  y_test_char = [char_classes_map[e] for e in y_test]

  train_mask = np.isin(y_train_char, legal_classes)
  y_train_char = np.asarray(y_train_char)
  y_train_char_subset = y_train_char[train_mask]
  Y_train = [num_classes_map[e] for e in y_train_char_subset]
  X_train = x_train[train_mask]
  print(len(X_train), len(Y_train), X_train.shape)
  #Y_test = [num_classes_map[e] for e in y_train_char]
  num_classes = len(id_classes) + 1
  X_train_exp, Y_train_cat = one_hot(X_train, Y_train, num_classes-1) # one_hot creates one more class
  img_size = X_train_exp[0].shape

  return char_classes_map, num_classes_map, id_classes, train_ood_classes, test_ood_classes, img_size, num_classes, x_test, y_test, X_train_exp, Y_train_cat


def ood_make_classesrand145():
  # the MNIST numbers set has 10 classes, we use characters to temporarily rename them
  char_classes_map = { 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j' }

  num_id_classes = 1
  num_ood_classes = 4

  seq = list(range(10))
  id_classes = []
  for i in range(num_id_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    id_classes.append(char_classes_map[choice])
  train_ood_classes = []
  for i in range(num_ood_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    train_ood_classes.append(char_classes_map[choice])
  test_ood_classes = list(char_classes_map.values())
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', num_id_classes, 'num_ood_classes', num_ood_classes)
  print(id_classes, train_ood_classes, test_ood_classes)

  id_class_labels = [x for x in range(len(id_classes))]
  train_ood_class_labels = np.ones(len(train_ood_classes))*len(id_classes)
  test_ood_class_labels = np.ones(len(test_ood_classes))*len(id_classes)

  print(id_class_labels, train_ood_class_labels, test_ood_class_labels)

  keys = np.concatenate((id_classes, train_ood_classes, test_ood_classes))
  print('keys', keys)
  values = np.concatenate((id_class_labels, train_ood_class_labels, test_ood_class_labels))
  print('values', values)
  assert len(keys) == len(values)
  num_classes_map = {}
  for i in range(len(keys)):
    num_classes_map[keys[i]] = int(values[i])
  print(num_classes_map)

  legal_classes = id_classes + train_ood_classes

  # get the raw mnist data and label classes as alphabetical characters
  x_train, y_train, x_test, y_test = get_raw_data()
  y_train_char = [char_classes_map[e] for e in y_train]
  y_test_char = [char_classes_map[e] for e in y_test]

  train_mask = np.isin(y_train_char, legal_classes)
  y_train_char = np.asarray(y_train_char)
  y_train_char_subset = y_train_char[train_mask]
  Y_train = [num_classes_map[e] for e in y_train_char_subset]
  X_train = x_train[train_mask]
  print(len(X_train), len(Y_train), X_train.shape)
  #Y_test = [num_classes_map[e] for e in y_train_char]
  num_classes = len(id_classes) + 1
  X_train_exp, Y_train_cat = one_hot(X_train, Y_train, num_classes-1) # one_hot creates one more class
  img_size = X_train_exp[0].shape

  return char_classes_map, num_classes_map, id_classes, train_ood_classes, test_ood_classes, img_size, num_classes, x_test, y_test, X_train_exp, Y_train_cat


def ood_make_classesrand235():
  # the MNIST numbers set has 10 classes, we use characters to temporarily rename them
  char_classes_map = { 0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j' }

  num_id_classes = 2
  num_ood_classes = 3

  seq = list(range(10))
  id_classes = []
  for i in range(num_id_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    id_classes.append(char_classes_map[choice])
  train_ood_classes = []
  for i in range(num_ood_classes):
    choice = rn.choice(seq)
    seq.remove(choice)
    train_ood_classes.append(char_classes_map[choice])
  test_ood_classes = list(char_classes_map.values())
  for cl in id_classes:
    test_ood_classes.remove(cl)
  for cl in train_ood_classes:
    test_ood_classes.remove(cl)

  print('num_id_classes', num_id_classes, 'num_ood_classes', num_ood_classes)
  print(id_classes, train_ood_classes, test_ood_classes)

  id_class_labels = [x for x in range(len(id_classes))]
  train_ood_class_labels = np.ones(len(train_ood_classes))*len(id_classes)
  test_ood_class_labels = np.ones(len(test_ood_classes))*len(id_classes)

  print(id_class_labels, train_ood_class_labels, test_ood_class_labels)

  keys = np.concatenate((id_classes, train_ood_classes, test_ood_classes))
  print('keys', keys)
  values = np.concatenate((id_class_labels, train_ood_class_labels, test_ood_class_labels))
  print('values', values)
  assert len(keys) == len(values)
  num_classes_map = {}
  for i in range(len(keys)):
    num_classes_map[keys[i]] = int(values[i])
  print(num_classes_map)

  legal_classes = id_classes + train_ood_classes

  # get the raw mnist data and label classes as alphabetical characters
  x_train, y_train, x_test, y_test = get_raw_data()
  y_train_char = [char_classes_map[e] for e in y_train]
  y_test_char = [char_classes_map[e] for e in y_test]

  train_mask = np.isin(y_train_char, legal_classes)
  y_train_char = np.asarray(y_train_char)
  y_train_char_subset = y_train_char[train_mask]
  Y_train = [num_classes_map[e] for e in y_train_char_subset]
  X_train = x_train[train_mask]
  print(len(X_train), len(Y_train), X_train.shape)
  #Y_test = [num_classes_map[e] for e in y_train_char]
  num_classes = len(id_classes) + 1
  X_train_exp, Y_train_cat = one_hot(X_train, Y_train, num_classes-1) # one_hot creates one more class
  img_size = X_train_exp[0].shape

  return char_classes_map, num_classes_map, id_classes, train_ood_classes, test_ood_classes, img_size, num_classes, x_test, y_test, X_train_exp, Y_train_cat


def ood_make_cases145(char_classes_map):
  all_classes = list(char_classes_map.values())
  id_comb = combinations(all_classes, 1)
  cases = []
  for id_cl in id_comb:
    this_classes = all_classes.copy()
    this_classes.remove(id_cl[0])
    train_ood_comb = combinations(this_classes, 4)
    for train_ood_cl in train_ood_comb:
      cases.append((id_cl,train_ood_cl))
  return cases


def ood_make_cases235(char_classes_map):
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



def analyze_results2(num_classes, char_classes_map, id_classes, train_ood_classes, test_ood_classes, y_test, y_pred):
  inv_char_classes_map = {v: k for k, v in char_classes_map.items()}
  id_res = []
  train_ood_res = []
  test_ood_res = []
#  for cl in range(10):
#    char = char_classes_map[cl]
  for char in np.concatenate((id_classes, train_ood_classes, test_ood_classes)):
    cl = inv_char_classes_map[char]
    #print(char, 'becomes', cl)
    mask = np.isin(y_test, cl)
    masked_y_pred = y_pred[mask]
    nums = [0]*num_classes
    for i in range(len(masked_y_pred)):
      nums[masked_y_pred[i]] += 1
    if char in id_classes:
      id_res.append((char, cl, nums))
    if char in train_ood_classes:
      train_ood_res.append((char, cl, nums))
    if char in test_ood_classes:
      test_ood_res.append((char, cl, nums))
  return id_res, train_ood_res, test_ood_res
  
def analyze_results(char_classes_map, id_classes, train_ood_classes, test_ood_classes, y_test, y_pred):
  in_val = 0
  in_count = 0
  out_val = 0
  out_count = 0
  test_val = 0
  test_count = 0
  res = []
  for cl in range(10):
    char = char_classes_map[cl]
    corr=0
    cs=0
    for i in range(len(y_pred)):
      if y_test[i] == cl:
        cs=cs+1
        if y_pred[i] == len(id_classes):
          corr=corr+1
    if char in id_classes:
      in_count = in_count + 1
      in_val = in_val + corr/cs
      res.append(('  id', cl, corr/cs))
    if char in train_ood_classes:
      out_count = out_count + 1
      out_val = out_val + corr/cs
      res.append((' ood', cl, corr/cs))
    if char in test_ood_classes:
      test_count = test_count + 1
      test_val = test_val + corr/cs
      res.append(('test', cl, corr/cs))
  return res, in_val / in_count, out_val / out_count, test_val / test_count

def get_raw_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255
  return x_train, y_train, x_test, y_test

def split_data(x, y):
  uniq = np.unique(y)
  assert len(uniq) == 10
  retx = [0]*len(uniq)
  rety = [0]*len(uniq)
  for cl in uniq:
    mask = np.isin(y, cl)
    x_cl, y_cl = x[mask], y[mask]
    retx[cl] = x_cl
    rety[cl] = y_cl
  return retx, rety

def get_raw_data_as_classes():
  x_train, y_train, x_test, y_test = get_raw_data()
  x_tr, y_tr = split_data(x_train, y_train)
  x_te, y_te = split_data(x_test, y_test)
  return x_tr, y_tr, x_te, y_te


def get_class(x, y, c):
  inclass = np.isin(y, c)
  return x[inclass], y[inclass]

def confusion_matrix_as_percent(confusion_matrix):
  print(confusion_matrix)
  n = confusion_matrix.numpy()
  p = n.sum(axis=1)
  for i in range(len(p)):
    print(i, n[i][i]/p[i])
  return

def calc_percentages(num_classes, id_classes, train_ood_classes, test_ood_classes, y_pred, y_test):
  id_val = 0
  id_count = 0
  train_ood_val = 0
  train_ood_count = 0
  test_ood_val = 0
  test_ood_count = 0
  res = []
  for cl in range(10):
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

def pred_percentage(y, pred, new, ood):
  corr=0
  cs=0
  for i in range(len(pred)):
    if y[i]==new:
      cs=cs+1
      if pred[i]==ood:
        corr=corr+1
  print(new, corr/cs)
  return(cs, corr)

def setup2(num_train, id_classes, train_ood_classes, test_ood_classes):
  img_size = (28, 28, 1)
  x_train, y_train, x_test, y_test = get_raw_data()
  id_label = [x for x in range(len(id_classes))]
  ood_label = len(id_classes)
  num_classes = ood_label + 1

  train_classes = id_clases + train_ood_classes
  train_mask = np.isin(y_train, train_classes)
  test_ood_mask = np.isin(y_train, test_ood_classes)
  X_train, Y_train = x_train[train_mask], y_train[train_mask]
  #### THIS WAS NOT FINISHED

  
def setup(num_train, num_holdout, id_classes, train_ood_classes, test_ood_classes):
  img_size = (28, 28, 1)
  #id_classes = [0, 1]
  #train_ood_classes = [2, 3, 4]
  #test_ood_classes = [5, 6, 7, 8, 9]

  x_train, y_train, x_test, y_test = get_raw_data()

  # use subset of the 10 MNIST classes
  # split training images to: train, holdout, unused
  id_label=[x for x in range(len(id_classes))]
  i=0
  for i in range(len(id_classes)):
    id_label[i]=i
  ood_label = len(id_classes) # this should be 2
  train_classes = id_classes + train_ood_classes
  all_legal_classes = id_classes + test_ood_classes + train_ood_classes
  num_classes = len(id_classes) + 1
  train_mask = np.isin(y_train, train_classes)
  test_mask = np.isin(y_test, all_legal_classes)
  X_train, Y_train = x_train[train_mask][0:num_train], y_train[train_mask][0:num_train]
  X_holdout, Y_holdout = x_train[train_mask][num_train:num_train+num_holdout], y_train[train_mask][num_train:num_train+num_holdout]
  X_test, Y_test = x_test[test_mask], y_test[test_mask]

  # Consider this list comprehension way of replacing 3, 4, 5, 6 with 2
  # instead of using the renumber_y subroutine
  print(np.mean(Y_train), np.mean(Y_holdout), np.mean(Y_test))
  Y_train[:] = [x if x in id_classes else ood_label for x in Y_train]
  Y_holdout[:] = [x if x in id_classes else ood_label for x in Y_holdout]
  Y_test[:] = [x if x in id_classes else ood_label for x in Y_test]
  print(np.mean(Y_train), np.mean(Y_holdout), np.mean(Y_test))
  return img_size, num_classes, X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, x_test, y_test
#img_size, num_classes, X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, x_test, y_test=setup(1000, 3000, [0,1], [2,3,4],[5,6,7,8,9])
def exp_cat(num_classes, X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test):

  # Convert the data to a format compatible with Keras (which expects additional dim)
  X_train_exp = np.expand_dims(X_train, -1)
  X_holdout_exp = np.expand_dims(X_holdout, -1)
  X_test_exp = np.expand_dims(X_test, -1)
  print("X_train_exp shape:", X_train_exp.shape, "X_holdout_exp shape", X_holdout_exp.shape, "X_test_exp shape", X_test_exp.shape)
  print("number of samples for train: ", X_train_exp.shape[0], "holdout: ", X_holdout_exp.shape[0], "test: ", X_test_exp.shape[0])
  Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)
  Y_holdout_cat = tf.keras.utils.to_categorical(Y_holdout, num_classes)
  Y_test_cat = tf.keras.utils.to_categorical(Y_test, num_classes)
  print('Y_train_cat shape;', Y_train.shape, "Y_holdout_cat shape", Y_holdout_cat.shape, "Y_test shape", Y_test_cat.shape)

  return X_train_exp, Y_train_cat, X_holdout_exp, Y_holdout_cat, X_test_exp, Y_test_cat


# The CNN
initial_learning_rate = 0.01
def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)

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

def train(model, X_train, Y_train):
  batch_size = 60
  epochs = 15
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  hist = model.fit(X_train, Y_train, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return hist

def train_small_condor(model, X_train, Y_train):
  batch_size = 60
  epochs = 3
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  hist = model.fit(X_train, Y_train, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return hist

def trainKL(model, X_train, Y_train):
  batch_size = 60
  epochs = 15
  model.compile(loss=tf.keras.losses.KLDivergence(), optimizer="adam", metrics=["accuracy"])
  hist = model.fit(X_train, Y_train, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return hist

def trainJSD(model, X_train, Y_train):
  batch_size = 60
  epochs = 40 # 15
  model.compile(loss=JSD_loss, optimizer="adam", metrics=["accuracy"])
  hist = model.fit(X_train, Y_train, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return hist

def JSD_loss(y_true, y_pred):
  m = 0.5 * (y_true + y_pred)
  return 0.5 * tf.keras.losses.kullback_leibler_divergence(y_true, m) + 0.5 * tf.keras.losses.kullback_leibler_divergence(y_pred, m)

# from KDD2020 Tutorial presented by nplan

# complete this function to calculate ece
def ece_calculation_binary(prob_true, prob_pred, bin_sizes):
    ### YOUR CODE HERE
    ece = 0
    for m in np.arange(len(bin_sizes)):
        ece = ece + (bin_sizes[m] / sum(bin_sizes)) * np.abs(prob_true[m] - prob_pred[m])
    return ece

# complete this function to calculate mce
def mce_calculation_binary(prob_true, prob_pred, bin_sizes):
    ### YOUR CODE HERE 
    mce = 0
    for m in np.arange(len(bin_sizes)):
        mce = max(mce, np.abs(prob_true[m] - prob_pred[m]))
    return mce

# complete this function to calculate rmsce
def rmsce_calculation_binary(prob_true, prob_pred, bin_sizes):
    ### YOUR CODE HERE 
    rmsce = 0
    for m in np.arange(len(bin_sizes)):
        rmsce = rmsce + (bin_sizes[m] / sum(bin_sizes)) * (prob_true[m] - prob_pred[m]) ** 2
    return np.sqrt(rmsce)

def plot_reliability_diagram(prob_true, prob_pred, model_name, ax=None):
    # Plot the calibration curve for ResNet in comparison with what a perfectly calibrated model would look like
    if ax==None:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
    else:
        plt.sca(ax)
    
    plt.plot([0, 1], [0, 1], color="#FE4A49", linestyle=":", label="Perfectly calibrated model")
    plt.plot(prob_pred, prob_true, "s-", label=model_name, color="#162B37")

    plt.ylabel("Fraction of positives", fontsize=16)
    plt.xlabel("Mean predicted value", fontsize=16,)

    plt.legend(fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, color="#B2C7D9")

    plt.tight_layout()

def ece_calculation_multiclass(y_true, y_pred):
    ### use calibration_curve and your binary function to complete this function
    ece_bin = []
    for a_class in range(y_true.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
        plot_reliability_diagram(prob_true, prob_pred, "Class {0}".format(a_class))
        bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
        ece_bin.append(ece_calculation_binary(prob_true, prob_pred, bin_sizes))
    ## here we have a choice - do we wish to weight our metric depending on the number
    ## of positive examples in each class, or take an unweighted mean
    
    # return sum(ece_bin*class_weights)/n_classes
    return np.mean(ece_bin)
        
    
def mce_calculation_multiclass(y_true, y_pred):
    ### use calibration_curve and your binary function to complete this function
    mce_bin = []
    for a_class in range(y_true.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
        bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
        mce_bin.append(mce_calculation_binary(prob_true, prob_pred, bin_sizes))
    ## here we have a choice - do we wish to weight our metric depending on the number
    ## of positive examples in each class, or take an unweighted mean
    
    # return sum(ece_bin*class_weights)/n_classes
    return np.mean(mce_bin)
    
def rmsce_calculation_multiclass(y_true, y_pred):
    ### use calibration_curve and your binary function to complete this function
    rmsce_bin = []
    for a_class in range(y_true.shape[1]):
        prob_true, prob_pred = calibration_curve(y_true[a_class], y_pred[a_class], n_bins=10)
        bin_sizes = np.histogram(a=y_pred[a_class], range=(0, 1), bins=len(prob_true))[0]
        rmsce_bin.append(rmsce_calculation_binary(prob_true, prob_pred, bin_sizes))
    ## here we have a choice - do we wish to weight our metric depending on the number
    ## of positive examples in each class, or take an unweighted mean
    
    # return sum(ece_bin*class_weights)/n_classes
    return np.mean(rmsce_bin)

# From KDD2020 calibration tutorial

# they did a binary and multiclass example but our model is multiclass
def calib(multiclass_model, multiclass_model_num_classes, x_val, y_val, x_test, y_test):
  #multiclass_model = model
  #multiclass_model_num_classes = num_classes
  #x_val = X_holdout
  #y_val = Y_holdout
  #x_test = X_test
  #y_test = Y_test
  # so renaming in above line to be able to use their code as is



  y_pred = multiclass_model.predict(x_val)
  acc_score = accuracy_score(np.argmax(y_val,1),np.argmax(y_pred,1))
  loss_score = log_loss(y_val, y_pred)
  print('Multiclass metrics: validation accuracy is {0:.2f}, validation loss is {1:.2f}'.format(acc_score, loss_score))

  # copy the model
  new_model = multiclass_model

  # get the tensor input to the final dense layer of the model
  pre_dense_out = new_model.layers[-2].output

  # reapply a final Dense layer - but this time with no softmax activation
  # set its weights to match the old model's dense layers
  pre_soft_out = tf.keras.layers.Dense(multiclass_model_num_classes, activation=None)(pre_dense_out)
  new_model = tf.keras.Model(inputs=new_model.input, outputs=pre_soft_out)
  new_model.layers[-1].set_weights(multiclass_model.layers[-1].get_weights())

  # we need to compile the model to predict from it
  new_model.compile(optimizer="Adam",
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

  y_logit = new_model.predict(x_val)

  def scale_fun_ce(x, *args):
    #Returns the NLL of the model over the validation set when scaled by the t parameter
    t = x[0]
    y_logit_scaled = y_logit/t
    y_pred_inner = softmax(y_logit_scaled, axis=1)
    return log_loss(y_val, y_pred_inner)


  min_obj = minimize(scale_fun_ce,[1],method='Nelder-Mead',options={'xatol': 1e-13, 'disp': True})
  print('calibration optimization result', min_obj.x[0])

  # From KDD2020 calibration tutorial

  # evaluate calibration on test set
  y_logit_test = new_model.predict(x_test)
  y_test_pred = multiclass_model.predict(x_test)

  # use learned scaling param to scale logits, and apply softmax
  temp_scaled = y_logit_test/min_obj.x[0]
  y_pred_test_corr = softmax(temp_scaled, axis=1)

  # plot pre-calibration reliability diag
  prob_true, prob_pred = calibration_curve(y_test.flatten(), y_test_pred.flatten(), n_bins=10)
  plot_reliability_diagram(prob_true, prob_pred, "All softmax outs - uncalibrated")
  bin_sizes = np.histogram(a=y_pred.flatten(), range=(0, 1), bins=10)[0]
  print("Uncal. RMSCE: ", rmsce_calculation_binary(prob_true, prob_pred, bin_sizes))

  # plot post-calibration reliability diag
  prob_true, prob_pred = calibration_curve(y_test.flatten(), y_pred_test_corr.flatten(), n_bins=10)
  plot_reliability_diagram(prob_true, prob_pred, "All softmax outs - calibrated")
  bin_sizes = np.histogram(a=y_pred.flatten(), range=(0, 1), bins=10)[0]
  print("Calib. RMSCE: ", rmsce_calculation_binary(prob_true, prob_pred, bin_sizes))

  return temp_scaled, min_obj.x[0], new_model

def find_new_class(input_index, y_pred, c, max_thres, min_thres):
  i=0
  index=[]
  #get all training imgs and add the twos from the test imgs to it
  while i<len(y_pred):
    if i in input_index:
      if y_pred[i,c] < max_thres and y_pred[i,c] > min_thres:
        index.append(i)
    i=i+1
  return index

def make_new_class(index, x_test, X_train, Y_train, c):
  for i in index:
    #X_test1 = np.expand_dims(x_test[i], axis=0)
    p=x_test[i]
    X = p[np.newaxis, :, :]
    print(X_train.shape, X.shape)
    X_train = np.vstack((X_train, X))
    Y_train = np.append(Y_train, [c])
  return X_train, Y_train
def addnewclass(ind, x_test, Y_test, X_train, Y_train, c):
  i=0
  j=0
  for i in ind:
    p=x_test[i]
    X = p[np.newaxis, :, :]
    print(X_train.shape, X.shape)
    X_train = np.vstack((X_train, X))
    Y_train = np.append(Y_train, [c])
    i=i+1
  return(X_train, Y_train)
def expand(X_train):
  #Convert the data to a format compatible with Keras (which expects additional dim)
  X_train_exp = np.expand_dims(X_train, -1)
  return(X_train_exp)
def one_hot(X_train, Y_train, c):
  num_classes = c+1
  X_train_exp = np.expand_dims(X_train, -1)
  Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)
  return(X_train_exp, Y_train_cat)

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

def print_count(ind, y):
  nums = [0]*10
  for i in range(len(ind)):
    nums[(int(y[ind[i]]))] += 1
  print(nums)

def print_count_noind(y):
  nums = [0]*10
  for i in range(len(y)):
    nums[int(y[i])] += 1
  print(nums)

