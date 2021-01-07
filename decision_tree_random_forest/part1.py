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

def get_attr_node(X_val, y_val, N, d):
	max_val = -100
	max_idx = -1

	for idx in range(d):
		col_val = X_val[:,idx]
		information_gain, feat_entropy = get_information_gain(col_val, y_val)

		if information_gain >= max_val:
			max_val = information_gain
			max_idx = idx

	return max_idx, max_val

def all_same(items):
    return all(x == items[0] for x in items)

def draw_tree(X_val, y_val, tree_node,max_depth, depth, features):
	
	if len(y_val) == 0:
		return None
	elif depth >= max_depth:
		return None #{'leaf' : True}
	elif all_same(y_val):
		return {'label': y_val[0]}
	else:
		N = X_val.shape[0]
		d = X_val.shape[1]

		att_idx, info_gain = get_attr_node(X_val, y_val, N, d)
		
		y_left = y_val[X_val[:,att_idx] == 0]
		y_right = y_val[X_val[:,att_idx] == 1]
		x_left = X_val[X_val[:,att_idx] == 0]
		x_right = X_val[X_val[:,att_idx] == 1]
		
		ly_z, ly_zidx, ly_n, ly_nidx = get_zero_nonzero(y_left)
		ry_z, ry_zidx, ry_n, ry_nidx = get_zero_nonzero(y_right)
			
		z, i, n, ii = get_zero_nonzero(y_val)
		if z > n:
			temp = 0
		else:
			temp = 1
			
		tree_node = {'node_index' : att_idx, 'node' : features[att_idx], 
		'leaf' : False, 'information_gain' : info_gain, 'depth' : depth, 'label' : temp}

		depth += 1
		tree_node['left'] = draw_tree(x_left, y_left, {}, max_depth, depth, features)
		tree_node['right'] = draw_tree(x_right, y_right, {}, max_depth, depth, features)

		return tree_node


def prediction(tree, row):
	cur_layer = tree
	while (cur_layer.get('leaf') == False):	
		labels.append(cur_layer['label'])
		if row[cur_layer['node_index']] == 0 :
			next_layer = cur_layer['left']
		else:
			next_layer = cur_layer['right']

		if next_layer == None:
			return cur_layer.get('label')
		else:
			cur_layer = next_layer

	else:
		return cur_layer.get('label')

def y_prediction(X_val, tree_node):
	N = X_val.shape[0]
	result = np.zeros(N)

	for i in range(N):
		result[i] = prediction(tree_node, X_val[i])
	return result



np.set_printoptions(threshold=np.inf)

def cal_accuracy(pred_y, true_y):
	predict = (true_y == pred_y)
	acc = (float) (np.count_nonzero(predict)) / len(true_y)

	return acc * 100


features = t_x.columns

import pprint
pp = pprint.PrettyPrinter(indent = 4)

d_max = [2, 5, 10, 20, 25, 30, 35, 40, 45, 50]
tree_node = draw_tree(tx, ty, {}, 30, 0, features)

training_accuracy = []
validation_accuracy = []

for dmax in d_max:
	tree_node = draw_tree(tx, ty, {}, dmax, 0, features)

	t_pred = y_prediction(tx, tree_node)
	t_acc = cal_accuracy(t_pred, ty)
	training_accuracy.append(t_acc)

	v_pred = y_prediction(dx, tree_node)
	v_acc = cal_accuracy(v_pred, dy)
	validation_accuracy.append(v_acc)


	print("training accuracy with depth: " +str(dmax) +" is : " + str(t_acc))
	print("validation accuracy with depth: " + str(dmax) + " is : " +str(v_acc))


tacc = dict(zip(d_max, training_accuracy))
vacc = dict(zip(d_max, validation_accuracy))
t = pd.DataFrame(tacc, index = [0])
v = pd.DataFrame(vacc, index = [0])

#tt = t.to_csv(r'/scratch/ohda/web/Codes/CS534/HW4/part1_trainacc.csv')
#vv = v.to_csv(r'/scratch/ohda/web/Codes/CS534/HW4/part1_validacc.csv')
