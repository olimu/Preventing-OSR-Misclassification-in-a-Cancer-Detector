import os, sys, glob
import argmaxish_thres as thres
import numpy as np
import pandas as pd

def process_run(run, x_test, y_test, train_dataset, type, given_thres):
  if '%' in run:
    thisrun, rest = run.split('%')
    id, ood = thisrun.split('-')
  else:
    id, ood = run.split('-')
  prefix, id0, id1 = id.split('_')
  ood0, ood1, ood2 = ood.split('_')
  id_classes = (id0, id1)
  id_classes = [int(e) for e in id_classes]
  train_ood_classes = (ood0, ood1, ood2)
  train_ood_classes = [int(e) for e in train_ood_classes]

  if run[0:3] == 'med':
    test_ood_classes = med_utils.make_test_ood_classes(id_classes, train_ood_classes)
    x_test_exp = med_utils.expand(x_test)
    model = med_utils.tf.keras.models.load_model(run)
    X_train, Y_train = med_utils.med_use_classes_unbalanced(id_classes, train_ood_classes, train_dataset)
    med_utils.med_print_count_noind(Y_train)
    Y_train_orig = Y_train.copy()
    assert(len(id_classes) == 2)
    mask = np.isin(Y_train_orig, id_classes[0])
    Y_train[mask] = 0
    mask = np.isin(Y_train_orig, id_classes[1])
    Y_train[mask] = 1
    for j in range(len(train_ood_classes)):
      mask = np.isin(Y_train_orig, train_ood_classes[j])
      Y_train[mask] = 2
    med_utils.med_print_count_noind(Y_train)
    '''
    if 'control' in run:
      threshold = 0.9
    else:
      threshold = med_utils.calc_threshold(X_train, Y_train, model, len(id_classes)+1, type)
      threshold = 0.1
    '''
    threshold = given_thres
  else:
    test_ood_classes = ood_utils.make_test_ood_classes(id_classes, train_ood_classes)
    x_test_exp = ood_utils.expand(x_test)
    model = ood_utils.tf.keras.models.load_model(run)
    X_train, Y_train, ignore = ood_utils.ood_use_classes_balanced(id_classes, train_ood_classes)
    ood_utils.print_count_noind(Y_train)
    Y_train_orig = Y_train.copy()
    assert(len(id_classes) == 2)
    mask = np.isin(Y_train_orig, id_classes[0])
    Y_train[mask] = 0
    mask = np.isin(Y_train_orig, id_classes[1])
    Y_train[mask] = 1
    for j in range(len(train_ood_classes)):
      mask = np.isin(Y_train_orig, train_ood_classes[j])
      Y_train[mask] = 2
    ood_utils.print_count_noind(Y_train)
    '''
    if 'control' in run:
      threshold = 0.8
    else:
      threshold = ood_utils.calc_threshold(X_train, Y_train, model, len(id_classes)+1, type)
      threshold = 0.1
    '''
    threshold = given_thres

  y_pred_test = model.predict(x_test_exp)
  if 'control' in run:
    data = np.concatenate((id_classes, train_ood_classes, test_ood_classes, np.squeeze(np.reshape(y_pred_test, (len(y_pred_test)*2, 1)))))
    return thres.call_thres_control(threshold, data, y_test)
  else:
    data = np.concatenate((id_classes, train_ood_classes, test_ood_classes, np.squeeze(np.reshape(y_pred_test, (len(y_pred_test)*3, 1)))))
    return thres.call_thres(threshold, data, y_test)

def find_runs(label, suffix):
  label = label + '_?_?-?_?_?' + suffix
  folders = glob.glob(label)
  for folder in folders:
    if os.path.isdir(folder):
      print(folder)
    else:
      print('error', folder)
      sys.exit()
  return folders

if __name__ == '__main__':
  if (len(sys.argv) == 2):
    if (sys.argv[1][0:3] == 'med') | (sys.argv[1][0:3] == 'ood'):
      if (sys.argv[1] == 'medrand') | (sys.argv[1] == 'medcontrol') | (sys.argv[1] == 'oodrand') | (sys.argv[1] == 'oodcontrol'):
        suffix = '%*'
      else:
        suffix = ''
      runs = find_runs(sys.argv[1], suffix)
    else:
      print('usage', sys.argv[0], 'med | ood')
  else:
    print('usage', sys.argv[0], 'med | ood')

  if len(runs) == 0:
    print('runs empty')
    sys.exit()

  if (sys.argv[1][0:3] == 'med'):
    import med_utils
    train_dataset, val_dataset, test_dataset = med_utils.med_get_datasets()
    x_test, y_test = med_utils.med_get_raw_data_from_dataset(test_dataset)
    x_val, y_val = med_utils.med_get_raw_data_from_dataset(val_dataset)
    testval = 'val'
    x_test = x_val
    y_test = y_val
  elif (sys.argv[1][0:3] == 'ood'):
    import ood_utils
    x_train, y_train, x_test, y_test = ood_utils.get_raw_data()
    train_dataset = False
    testval = ''
  else:
    print('errrror')

  df = pd.DataFrame(columns=['thres', 'n0', 'n1', 'ntr', 'nte', 'f0', 't0', 'f1', 't1', 'f2tr', 't2tr', 'f2te', 't2te', 'f2te0', 'f2te1','type'])
  print('n0', 'n1', 'ntr', 'nte', 'f0', 't0', 'f1', 't1', 'f2tr', 't2tr', 'f2te', 't2te', 'f2te0', 'f2te1')    
  for run in runs:
    type = 'mean'
    given_thres = 0.999
    threshold, n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1 = process_run(run, x_test, y_test, train_dataset, type, given_thres)
    df.loc[len(df.index)] = (threshold, n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1,type)
  df.to_csv(sys.argv[1]+testval+'df4.csv', index=False)

# 1 medcontrol
# 2 oodcontrol
# 3 medrand
# 4 oodrand


