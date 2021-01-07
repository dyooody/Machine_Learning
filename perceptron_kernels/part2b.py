import numpy as np
import time
import pandas as pd

t_x = pd.read_csv("pa3_train_X.csv")
t_y = pd.read_csv("pa3_train_y.csv")
d_x = pd.read_csv("pa3_dev_X.csv")
d_y = pd.read_csv("pa3_dev_y.csv")

temp = t_x['Intercept']
t_x = t_x.drop('Intercept', 1)
t_x.insert(0, 'Intercept', temp)

temp = d_x['Intercept']
d_x = d_x.drop('Intercept', 1)
d_x.insert(0, 'Intercept', temp)

p_values = [1, 2, 3, 4, 5]

def gram_matrix(X_val, X2_val, p):
	X = np.asarray(X_val)
	X2 = np.asarray(X2_val)

	K = np.power((1 + np.dot(X, X2.T)), p)
	return K

def calculate_acc(X_val, y_train, y_val, Kernal, alpha):
	N = X_val.shape[0]

	X = np.asarray(X_val)
	y = np.asarray(y_val)
	y = y.flatten()

	y_tr = np.asarray(y_train)
	y_tr = y_tr.flatten()
	
	predict = (y == np.sign(np.dot(np.multiply(alpha, y_tr), Kernal)))
	acc = (float) (np.count_nonzero(predict)) / len(y)
	
	return acc * 100

np.set_printoptions(threshold=np.inf)

def learning(t_x, t_y, d_x, d_y, max_itr, p):
	N = t_x.shape[0]
	Kernal = gram_matrix(t_x, t_x, p)
	VKernal = gram_matrix(t_x, d_x, p)

	alpha = np.zeros(N)
	
	u = np.zeros(N)

	X = np.asarray(t_x)
	y = np.asarray(t_y)
	y = y.flatten()

	X_v = np.asarray(d_x)
	y_v = np.asarray(d_y)
	y_v = y_v.flatten()

	itr_cnt = 0
	t = time.time()

	train_acc_vals = []
	valid_acc_vals = []

	
	while (itr_cnt < max_itr):
		itr_cnt += 1
		
		val = np.multiply(alpha.T, Kernal)
		u = np.dot(val, y)
		upd = np.multiply(u, y)

		alpha = alpha + 1 * np.where(upd <= 0, 1, 0)
		
		#for i in range(N):
		#	if upd[i] <= 0:
		#		alpha[i] = alpha[i] + 1
		
		train_acc = calculate_acc(t_x, t_y, t_y, Kernal, alpha)
		train_acc_vals.append(train_acc)
		
		valid_acc = calculate_acc(d_x, t_y, d_y, VKernal, alpha)
		valid_acc_vals.append(valid_acc)
		
		if(itr_cnt % 10 == 0):
			print("Train accuracy with p value "+ str(p) + " is : " + str(train_acc) + " itr_cnt : " + str(itr_cnt))
			print("Validation accuracy with p value " + str(p) + " is : " + str(valid_acc) + " itr_cnt : " + str(itr_cnt))
	
	elapsed = time.time() - t
	
	return (alpha, train_acc_vals, valid_acc_vals, elapsed)

max_itr = 100

alpha, train_acc_vals, valid_acc_vals, elapsed = learning(t_x, t_y, d_x, d_y, max_itr, 1)

'''
times = []

for itr in range(1, max_itr+1):
	alpha, train_acc_vals, valid_acc_vals, time_val = learning(t_x, t_y, d_x, d_y, itr, 1)
	times.append(time_val)
	if(itr % 10 == 0):
		print()
		print("time took : " + str(time_val) + " with p value : " + str(1) + " with iteration : " + str(itr))

index_list = list(range(1, max_itr+1))		
t = dict(zip(index_list, times))
tt = pd.DataFrame(t, index = [0])
tcsv = tt.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/pt2b_times_'+str(1)+'.csv')

'''
index_list = list(range(1, max_itr+1))		
tav = dict(zip(index_list, train_acc_vals))
dav = dict(zip(index_list, valid_acc_vals))

tavdf = pd.DataFrame(tav, index = [0])
davdf = pd.DataFrame(dav, index = [0])

#tavcsv = tavdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/pt2b_trainacc_'+str(1)+'.csv')
#davcsv = davdf.to_csv(r'/scratch/ohda/web/Codes/CS534/HW3/pt2b_validacc_'+str(1)+'.csv')
