import numpy as np
import pandas as pd
import math

t_x = pd.read_csv("pa4_train_X.csv")
t_y = pd.read_csv("pa4_train_y.csv", header = None)
d_x = pd.read_csv("pa4_dev_X.csv")
d_y = pd.read_csv("pa4_dev_y.csv", header = None)

tx = np.asarray(t_x)
ty = np.asarray(t_y).flatten()
dx = np.asarray(d_x)
dy = np.asarray(d_y).flatten()

np.set_printoptions(threshold=np.inf)

def get_random_value(X_val, y_val):
	
	N = X_val.shape[0]
	idx = np.random.choice(N, N, replace = True)
	rand_x = np.zeros(shape = X_val.shape)
	rand_y = np.zeros(shape = y_val.shape)
	for i in range(len(idx)):
		rand_x[i] = X_val[idx[i]]
		rand_y[i] = y_val[idx[i]]
	return rand_x, rand_y

def get_selected_values(X_val,fea):
	N = X_val.shape[0]
	sel_x = np.zeros(shape=(N, len(fea)))
	for i in range(len(fea)):
		sel_x[:,i] = X_val[:,fea[i]]

	return sel_x

def select_random_features(X_val, ex_fea, num_fea):
	d = X_val.shape[1]
	fea = []
	for j in range(num_fea):
		a = np.random.choice([i for i in range(0, d) if i not in fea])
		fea.append(a)

	#fea = np.random.choice(d, num_fea, replace = False)
	return fea, ex_fea

def get_zero_nonzero(feats):
	nonzero_val = np.count_nonzero(feats)
	nonzero_idx = np.nonzero(feats)
	zero_val = np.count_nonzero(feats == 0)
	zero_idx = np.nonzero(feats == 0)

	return zero_val, zero_idx, nonzero_val, nonzero_idx

def each_entropy(v, N):
	return -(v*1.0/N)*math.log(v*1.0/N, 2)

def each_info(v, N, ent):
	return (v*1.0/N)*ent

def entropy_val(v1, v2, N):
	if (v1 == 0 or v2 == 0):
		return 0
	val1 = each_entropy(v1, N)
	val2 = each_entropy(v2, N)
	return val1 + val2

def information_val(v1, ent1, v2, ent2, N):
	val1 = each_info(v1, N, ent1)
	val2 = each_info(v2, N, ent2)
	return val1 + val2

def get_information_gain(col_val, y_val):
	N = len(col_val)
	zero_val, zero_idx, nonzero_val, nonzero_idx = get_zero_nonzero(col_val)
	yz, yz_idx, ynz, ynz_idx = get_zero_nonzero(y_val)
	root_entropy = entropy_val(yz, ynz, len(y_val))

	#consider left branch as zero, right as nonzero
	left_y = np.take(y_val, zero_idx)
	ly_zero, ly_zero_idx, ly_non, ly_non_idx = get_zero_nonzero(left_y)
	left_entropy = entropy_val(ly_zero, ly_non, zero_val)
	
	right_y = np.take(y_val, nonzero_idx)
	ry_zero, ry_zero_idx, ry_non, ry_non_idx = get_zero_nonzero(right_y)
	right_entropy = entropy_val(ry_zero, ry_non, nonzero_val)
	
	feat_entropy = information_val(zero_val, left_entropy, nonzero_val, right_entropy, N)
	information_gain = root_entropy - feat_entropy

	return information_gain, feat_entropy

def get_attr_node(X_val, y_val, fea):
	max_val = -100
	max_idx = -1
	for idx in range(len(fea)):

		col_val = X_val[:,fea[idx]]
		information_gain, feat_entropy = get_information_gain(col_val, y_val)

		if information_gain >= max_val:
			max_val = information_gain
			max_idx = fea[idx]

	return max_idx, max_val

def all_same(items):
    return all(x == items[0] for x in items)

def draw_tree(X_val, y_val, tree_node, max_depth, depth, features, num_fea, excludes):

	if len(y_val) == 0:
		return None
	elif depth >= max_depth:
		return None #{'leaf' : True}
	elif all_same(y_val):
		return {'label': int(y_val[0])}
	else:
		fea, ex_fea = select_random_features(X_val, excludes, num_fea)
		att_idx, info_gain = get_attr_node(X_val, y_val, fea)
		ex_fea.append(att_idx)
			
		#y_left : where val = 0, y_right = where val = 1
		y_left = y_val[X_val[:,att_idx] == 0]
		y_right = y_val[X_val[:,att_idx] == 1]
		x_left = X_val[X_val[:,att_idx] == 0]
		x_right = X_val[X_val[:,att_idx] == 1]
		
		ly_z, ly_zidx, ly_n, ly_nidx = get_zero_nonzero(y_left)
		ry_z, ry_zidx, ry_n, ry_nidx = get_zero_nonzero(y_right)
			
		z, i, n, ii = get_zero_nonzero(y_val)
		if z >= n:
			label = 0
		else:
			label = 1
			
		tree_node = {'node_index' : att_idx, 'node' : features[att_idx], 
		'leaf' : False, 'information_gain' : info_gain, 'depth' : depth, 'label' : label}
		
		depth += 1
		tree_node['left'] = draw_tree(x_left, y_left, {}, max_depth, depth, features, num_fea, ex_fea)
		tree_node['right'] = draw_tree(x_right, y_right, {}, max_depth, depth, features, num_fea, ex_fea)

		return tree_node

def prediction(tree, row):
	cur_layer = tree
	while (cur_layer.get('leaf') == False):	
		if row[cur_layer['node_index']] == 0:
			next_layer = cur_layer['left']
		else:
			next_layer = cur_layer['right']

		if next_layer == None:
			return cur_layer.get('label')
		else:
			cur_layer = next_layer

	else:
		return cur_layer.get('label')

def y_prediction(X_val, forests):
	N = X_val.shape[0]
	result = np.zeros(N)

	for i in range(N):
		vals = np.zeros(len(forests))
		for j in range(len(forests)):
			val = prediction(forests[j], X_val[i])
			vals[j] = val

		zero = np.count_nonzero(vals == 0)
		nonzero = np.count_nonzero(vals)
		if zero >= nonzero:
			result[i] = 0
		else:
			result[i] = 1
		
	return result


np.set_printoptions(threshold=np.inf)

def cal_accuracy(pred_y, true_y):
	
	predict = (true_y == pred_y)
	acc = (float) (np.count_nonzero(predict)) / len(true_y)

	return acc * 100


features = t_x.columns

# num_fea = m
# T: the number of trees to include in your random forest
# m : the number of features to sub-sample in each test selection step
# dmax : the maximum depth of the tress in your random forest


d_max = [2, 10, 25]
T = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
m_fea = [5, 25, 50, 100]

depth = 0

for dmax in d_max:
	print("--------------------------------------------------------")
	for m in m_fea:
		print("=========================================================")
		forests = []
		np.random.seed(1)

		for i in range(100):
			ex_fea = []
			rand_x, rand_y = get_random_value(tx, ty)
			tree_node = draw_tree(rand_x, rand_y, {}, dmax, depth, features, m, ex_fea)
			forests.append(tree_node)

		train_accs = []
		valid_accs = []	

		for t in T:
			t_result = y_prediction(tx, forests[:t])
			t_acc = cal_accuracy(t_result, ty)
			v_result = y_prediction(dx, forests[:t])
			v_acc = cal_accuracy(v_result, dy)
			print("train accuracy: d_max : " + str(dmax) + ", m : " +str(m)+ " with tree num : " + str(t) +" = " + str(t_acc) )
			print("valid accuracy: d_max : " + str(dmax) + ", m : " +str(m)+ " with tree num : " + str(t) +" = " + str(v_acc) )
			
			train_accs.append(t_acc)
			valid_accs.append(v_acc)

		tacc = dict(zip(T, train_accs))
		vacc = dict(zip(T, valid_accs))

		t = pd.DataFrame(tacc, index = [0])
		v = pd.DataFrame(vacc, index = [0])

		#tt = t.to_csv(r'/scratch/ohda/web/Codes/CS534/HW4/'+str(dmax)+'_'+str(m)+'_trainacc.csv')
		#vv = v.to_csv(r'/scratch/ohda/web/Codes/CS534/HW4/'+str(dmax)+'_'+str(m)+'_validacc.csv')
		print()

