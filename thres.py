import numpy as np


# mid0, mid1 and mtr are masks for these classes
def calculate_given_thresholds(y, thres, mid0, mid1, mtr, mte):
  #print(len(y), len(mid0), len(mid1), len(mtr), len(mte))
  true0 = 0
  false0 = 0
  true1 = 0
  false1 = 0
  true2tr = 0
  false2tr = 0
  true2te = 0
  false2te = 0
  false2teid0 = 0
  false2teid1 = 0
  num_id0 = 0
  num_id1 = 0
  num_tr = 0
  num_te = 0
  for i in range(len(y)):
    if mid0[i]:
      num_id0 += 1
      if y[i][0] < thres:  
        false0 += 1
      else:
        true0 += 1
    elif mid1[i]:
      num_id1 += 1
      if y[i][1] < thres:
        false1 += 1
      else:
        true1 += 1
    elif mtr[i]:
      num_tr += 1
      if y[i][2] > thres:
        true2tr += 1
      else:
        false2tr += 1
    elif mte[i]:
      num_te += 1
      if y[i][2] > thres:
        true2te += 1
      else:
        false2te += 1
        if y[i][0] > thres:
          false2teid0 += 1
        if y[i][1] > thres:
          false2teid1 += 1
    else:
      print('error')
  assert(num_id0 + num_id1 + num_tr + num_te == len(y))
  return num_id0, num_id1, num_tr, num_te, false0, true0, false1, true1, false2tr, true2tr, false2te, true2te, false2teid0, false2teid1

# num_imgs is 7180 for med test, 10004 for med val or 10000 for ood test
# num_cats is 9 for med or 10 for ood
def call_thres(thres, data, y_test): 
  num_cats = len(np.unique(y_test))
  num_imgs = len(y_test)
  print('num_cats:', num_cats, 'num_imgs:', num_imgs)
  classes=data[0:num_cats]
  classes=[int(e) for e in classes]
  id_classes=classes[0:2]
  train_ood_classes=classes[2:5]
  mask0 = np.isin(y_test, train_ood_classes[0])
  mask1 = np.isin(y_test, train_ood_classes[1])
  mask2 = np.isin(y_test, train_ood_classes[2])
  train_ood_mask = mask0 | mask1 | mask2 
  test_ood_classes=classes[5:]
  mask0 = np.isin(y_test,test_ood_classes[0])
  mask1 = np.isin(y_test,test_ood_classes[1])
  mask2 = np.isin(y_test,test_ood_classes[2])
  mask3 = np.isin(y_test,test_ood_classes[3])
  if num_cats == 9:
    test_ood_mask = mask0 | mask1 | mask2 | mask3
  else:
    mask4 = np.isin(y_test,test_ood_classes[4])
    test_ood_mask = mask0 | mask1 | mask2 | mask3 | mask4

  y_pred=np.reshape(data[num_cats:], (num_imgs, 3))

  n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1 = calculate_given_thresholds(y_pred, thres, np.isin(y_test, id_classes[0]), np.isin(y_test, id_classes[1]), train_ood_mask, test_ood_mask)
  print(n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1)
  return thres, n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1

# mid0, mid1 and mtr are masks for these classes
def calculate_given_thresholds_control(y, thres, mid0, mid1, mtr, mte):
  #print(len(y), len(mid0), len(mid1), len(mtr), len(mte))
  true0 = 0
  false0 = 0
  true1 = 0
  false1 = 0
  true2tr = 0
  false2tr = 0
  true2te = 0
  false2te = 0
  false2teid0 = 0
  false2teid1 = 0
  num_id0 = 0
  num_id1 = 0
  num_tr = 0
  num_te = 0
  for i in range(len(y)):
    if mid0[i]:
      num_id0 += 1
      if y[i][0] < thres:  
        false0 += 1
      else:
        true0 += 1
    elif mid1[i]:
      num_id1 += 1
      if y[i][1] < thres:
        false1 += 1
      else:
        true1 += 1
    elif mtr[i]:
      num_tr += 1
      #if y[i][2] > thres:
      true2tr += 1
      #else:
      #  false2tr += 1
    elif mte[i]:
      num_te += 1
      #if y[i][2] > thres:
      true2te += 1
      #else:
      #  false2te += 1
      if y[i][0] > thres:
        false2teid0 += 1
      if y[i][1] > thres:
        false2teid1 += 1
    else:
      print('error')
  assert(num_id0 + num_id1 + num_tr + num_te == len(y))
  return num_id0, num_id1, num_tr, num_te, false0, true0, false1, true1, false2tr, true2tr, false2te, true2te, false2teid0, false2teid1

# num_imgs is 7180 for med test, 10004 for med val or 10000 for ood test
# num_cats is 9 for med or 10 for ood
def call_thres_control(thres, data, y_test): 
  num_cats = len(np.unique(y_test))
  num_imgs = len(y_test)
  print('num_cats:', num_cats, 'num_imgs:', num_imgs)
  classes=data[0:num_cats]
  classes=[int(e) for e in classes]
  id_classes=classes[0:2]
  train_ood_classes=classes[2:5]
  mask0 = np.isin(y_test, train_ood_classes[0])
  mask1 = np.isin(y_test, train_ood_classes[1])
  mask2 = np.isin(y_test, train_ood_classes[2])
  train_ood_mask = mask0 | mask1 | mask2 
  test_ood_classes=classes[5:]
  mask0 = np.isin(y_test,test_ood_classes[0])
  mask1 = np.isin(y_test,test_ood_classes[1])
  mask2 = np.isin(y_test,test_ood_classes[2])
  mask3 = np.isin(y_test,test_ood_classes[3])
  if num_cats == 9:
    test_ood_mask = mask0 | mask1 | mask2 | mask3
  else:
    mask4 = np.isin(y_test,test_ood_classes[4])
    test_ood_mask = mask0 | mask1 | mask2 | mask3 | mask4

  y_pred=np.reshape(data[num_cats:], (num_imgs, 2))

  n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1 = calculate_given_thresholds_control(y_pred, thres, np.isin(y_test, id_classes[0]), np.isin(y_test, id_classes[1]), train_ood_mask, test_ood_mask)
  print(n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1)
  return thres, n0, n1, ntr, nte, f0, t0, f1, t1, f2tr, t2tr, f2te, t2te, f2te0, f2te1

