import sys, csv, time
import numpy as np
import random as rn
import ood_utils
import thres


def run(id_classes, train_ood_classes):
  start = time.time()

  x_train, y_train, x_test, y_test = ood_utils.get_raw_data()

  X_train, Y_train, test_ood_classes = ood_utils.ood_use_classes_control(id_classes, train_ood_classes)
  ood_utils.print_count_noind(Y_train)

  Y_train_orig = Y_train.copy()
  assert(len(id_classes) == 2)
  mask = np.isin(Y_train_orig, id_classes[0])
  Y_train[mask] = 0
  mask = np.isin(Y_train_orig, id_classes[1])
  Y_train[mask] = 1
#  for j in range(len(train_ood_classes)):
#    mask = np.isin(Y_train_orig, train_ood_classes[j])
#    Y_train[mask] = 2
  ood_utils.print_count_noind(Y_train)

  num_classes = len(id_classes) # +1
  X_train_exp, Y_train_cat = ood_utils.one_hot(X_train, Y_train, num_classes-1) ### remember one-hot increments num_classes
  img_size = X_train_exp[0].shape


  model = ood_utils.setup_model(img_size, num_classes)


  print(X_train_exp.shape, Y_train_cat.shape, num_classes, img_size)
  hist = ood_utils.train(model, X_train_exp, Y_train_cat)
  print(hist.history.keys())

  threshold = ood_utils.calc_threshold(X_train, Y_train, model, num_classes, type='max')

  my_x_test_exp = ood_utils.expand(x_test)
  y_pred_test = model.predict(my_x_test_exp)
  y_pred_test_orig = y_pred_test.copy()
  y_pred_test_argmax = y_pred_test.argmax(axis=1)


  test_ood_classes = ood_utils.make_test_ood_classes(id_classes, train_ood_classes)
  id_val, id_count, train_ood_val, train_ood_count, test_ood_val, test_ood_count, res = ood_utils.calc_percentages(num_classes, id_classes, train_ood_classes, test_ood_classes, y_pred_test_argmax, y_test)
  end = time.time()
  print('run took', round(end-start), 's')

  id_name = '_'.join(str(e) for e in id_classes)
  train_ood_name = '_'.join(str(e) for e in train_ood_classes)
  name = id_name + '-' + train_ood_name

  #return id_val / id_count, train_ood_val / train_ood_count, test_ood_val / test_ood_count, id_classes, train_ood_classes, test_ood_classes, res, hist.history['loss'][len(hist.history['loss'])-1], hist.history['val_loss'][len(hist.history['val_loss'])-1], hist.history['accuracy'][len(hist.history['accuracy'])-1], hist.history['val_accuracy'][len(hist.history['val_accuracy'])-1], np.reshape(y_pred_orig, (30000,1))
  return model, name, threshold, np.concatenate((id_classes, train_ood_classes, test_ood_classes, np.squeeze(np.reshape(y_pred_test_orig, (len(y_pred_test_orig)*2,1)))))

def num_images(threshold, data):
  mask = (data[:, 0] < threshold) & (data[:, 1] < threshold)
  newdata = data[mask]
  print(len(newdata))
  

if __name__ == '__main__':
  ood_utils.init_random_seeds(no_cuda = False) # sets the random

  x_train, y_train, x_test, y_test = ood_utils.get_raw_data()
  case = ((0, 1), (2, 3, 4))
  id_classes = case[0]
  train_ood_classes = case[1]
  print(id_classes, train_ood_classes)
  for i in range(10):
    mod, name, threshold, dat = run(id_classes, train_ood_classes)
    mod.save('oodcontrol_' + name + '%' +str(time.time()-1675389656))
    print(len(dat))
    thres.call_thres_control(threshold, dat, y_test)
    num_images(threshold, np.reshape(dat[10:], (10000, 2)))
