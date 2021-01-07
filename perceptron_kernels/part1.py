import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt 

t_x = pd.read_csv("pa3_train_X.csv")
t_y = pd.read_csv("pa3_train_y.csv")
d_x = pd.read_csv("pa3_dev_X.csv")
d_y = pd.read_csv("pa3_dev_y.csv")

print(t_x.head())

'''
num_features = ['Age', 'Annual_Premium', 'Vintage']

train_numf_max = t_x[num_features].max()
train_numf_min = t_x[num_features].min()

dev_numf_max = d_x[num_features].max()
dev_numf_min = d_x[num_features].min()

train_norm = [];
for i in range(len(num_features)):
  norm_fea = [];
  cur_fea = num_features[i];
  range_val = train_numf_max[cur_fea] - train_numf_min[cur_fea];
  for j in range(len(t_x[cur_fea])):
    cur_val = t_x[cur_fea][j];
    norm = float((cur_val - train_numf_min[cur_fea]) / (range_val) )
    norm_fea.append(norm)
  train_norm.append(norm_fea)

train_norm = np.asarray(train_norm)
dict_train_norm = dict(zip(num_features, train_norm))

dev_norm = [];
for i in range(len(num_features)):
  norm_fea = [];
  cur_fea = num_features[i];
  range_val = dev_numf_max[cur_fea] - dev_numf_min[cur_fea];
  for j in range(len(d_x[cur_fea])):
    cur_val = d_x[cur_fea][j];
    norm = float((cur_val - dev_numf_min[cur_fea]) / (range_val) )
    norm_fea.append(norm)
  dev_norm.append(norm_fea)


dev_norm = np.asarray(dev_norm)
dict_dev_norm = dict(zip(num_features, dev_norm))

for i in range(len(num_features)):
  t_x[num_features[i]] = dict_train_norm[num_features[i]]
  d_x[num_features[i]] = dict_dev_norm[num_features[i]]
'''
#move Intercept column to first column
temp = t_x['Intercept']
t_x = t_x.drop('Intercept', 1)
t_x.insert(0, 'Intercept', temp)
print(t_x.head())

temp = d_x['Intercept']
d_x = d_x.drop('Intercept', 1)
d_x.insert(0, 'Intercept', temp)

def get_deter(x, y, w):
	inst = np.dot(w.T, x)
	#print(inst.shape)
	result = y * inst
	return result


def learning(X_val, y_val, max_itr):
	N = X_val.shape[0]
	d = X_val.shape[1]

	w = np.zeros(d)
	w_vals = []
	w_bar = np.zeros(d)
	w_bar_vals = []
	ex_cnt = 1

	X = np.asarray(X_val)
	print(X[1].shape)
	y = np.asarray(y_val)
	y = y.flatten()

	t = time.time()
	itr_cnt = 0;


	while (itr_cnt < max_itr):
		itr_cnt += 1
		if(itr_cnt % 10 == 0):
			print(itr_cnt)
		for i in range(N):
			detem = get_deter(X[i], y[i], w)
			if detem <= 0:
				w = w + (y[i] * X[i])
				
			w_bar = ((ex_cnt * w_bar) + w) / (ex_cnt + 1)
			ex_cnt += 1
			
		w_vals.append(w)
		w_bar_vals.append(w_bar)
	
	elapsed = time.time() - t
	return (w, w_vals, w_bar, w_bar_vals, elapsed)


def calculate_acc(X_val, y_val, weights):
	X = np.asarray(X_val)
	y = np.asarray(y_val)
	y = y.flatten()

	predict = (y == np.sign(np.dot(X_val, weights)))
	acc = (float)(np.count_nonzero(predict))/len(y)
	return acc * 100 

max_itr = 101
weights, weights_vals, weights_bar, weights_bar_vals, elapsed = learning(t_x, t_y, max_itr)
print("Time took : " + str(elapsed) + " with iteration : " + str(max_itr))
print(weights)
print(weights_bar)

# accuracy for 100th iteration 
weights_train_acc = calculate_acc(t_x, t_y, weights)
print("Accuracy of training data with online perceptron : " + str(weights_train_acc))
weights_bar_train_acc = calculate_acc(t_x, t_y, weights_bar)
print("Accuracy of training data with average perceptron : " + str(weights_bar_train_acc))

weights_dev_acc = calculate_acc(d_x, d_y, weights)
print("Accuracy of validation data with online perceptron : " + str(weights_dev_acc))
weights_bar_dev_acc = calculate_acc(d_x, d_y, weights_bar)
print("Accuracy of validation data with average perceptron : " + str(weights_bar_dev_acc))

print()

wg = dict(zip(list(t_x.columns), weights))
wgb = dict(zip(list(t_x.columns), weights_bar))

wdf = pd.DataFrame(wg, index= [0])
wbdf = pd.DataFrame(wgb, index = [0])

#wcsv = wdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/weights.csv')
#wbcsv = wbdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/weights_bar.csv')


# saving all weights based on the iteration 
weights_train_acc_vals = []
weights_bar_train_acc_vals = []
weights_dev_acc_vals = []
weights_bar_dev_acc_vals = []

for idx in range(1, max_itr):
	weights_train_acc = calculate_acc(t_x, t_y, weights_vals[idx])
	weights_train_acc_vals.append(weights_train_acc)
	weights_bar_train_acc = calculate_acc(t_x, t_y, weights_bar_vals[idx])
	weights_bar_train_acc_vals.append(weights_bar_train_acc)
	weights_dev_acc = calculate_acc(d_x, d_y, weights_vals[idx])
	weights_dev_acc_vals.append(weights_dev_acc)
	weights_bar_dev_acc = calculate_acc(d_x, d_y, weights_bar_vals[idx])
	weights_bar_dev_acc_vals.append(weights_bar_dev_acc)
	if( idx % 10 == 0):
		print("Accuracy of training data with online perceptron : " + str(weights_train_acc) +" with iteration : " + str(idx))
		print("Accuracy of training data with average perceptron : " + str(weights_bar_train_acc) +" with iteration : " + str(idx))
		print("Accuracy of validation data with online perceptron : " + str(weights_dev_acc) +" with iteration : " + str(idx))
		print("Accuracy of validation data with average perceptron : " + str(weights_bar_dev_acc) +" with iteration : " + str(idx))
		print()


index_list = list(range(1,max_itr+1))
wtav = dict(zip(index_list, weights_train_acc_vals))
wbtav = dict(zip(index_list, weights_bar_train_acc_vals))
wdav = dict(zip(index_list, weights_dev_acc_vals))
wbdav = dict(zip(index_list, weights_bar_dev_acc_vals))

wtavdf = pd.DataFrame(wtav, index=[0])
wbtavdf = pd.DataFrame(wbtav, index=[0])
wdavdf = pd.DataFrame(wdav, index=[0])
wbdavdf = pd.DataFrame(wbdav, index=[0])

#wtavcsv = wtavdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/accuracy_train_weight_iteration.csv')
#wbtavcsv = wbtavdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/accuracy_train_weight_bar_iteration.csv')
#wdavcsv = wdavdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/accuracy_dev_weight_iteration.csv')
#wbdavcsv = wbdavdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/accuracy_dev_weight_bar_iteration.csv')



