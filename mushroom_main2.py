import pandas as pd
import numpy as np
import matplotlib as plt


def entropy(df,target_attribute, Y):
	y_unique, y_count = np.unique(Y, return_counts=True)
	target_unique, target_count = np.unique(df[target_attribute], return_counts=True)
	splitcount_target = []
	prob_target = []
	#	calculating prob of each attribute w.r.t the output class
	unique_index = 0
	for att_classvalue in target_unique:
		count0,count1 = 0,0
		for target_value,y_value in zip(df[target_attribute],Y):
			if att_classvalue == target_value and y_value == y_unique[0]:
				count0 += 1
			elif att_classvalue == target_value and y_value == y_unique[1]:
				count1 += 1
		splitcount_target.append([count0,count1])
		prob_target.append([count0/target_count[unique_index],count1/target_count[unique_index]])
		unique_index += 1
	entropy_list = []
	for probs in prob_target:
		H = 0
		for p in probs:
			if p!= 0:
				H += -p*np.log2(p)
		entropy_list.append(H)
	print('\nUnique values:',target_unique)
	# print('entropy_list: ',entropy_list)
	print('splitcount_target: ',splitcount_target)

	# print('\ntotal rows:',df.shape[0])
	entropy_list = np.array(entropy_list)
	target_count = np.array(target_count)
	# print(entropy_list.shape,target_count.shape)
	entropy_list = entropy_list.reshape(entropy_list.shape[0],1)
	target_count = target_count.reshape(target_count.shape[0], 1)
	# print(entropy_list.shape, target_count.shape)
	final_ent = np.dot(entropy_list.T,target_count/df.shape[0])
	print('Probabilities: ',prob_target)
	print('Entropy List: ', entropy_list)
	print('final_ent',final_ent)

	# return
	return final_ent[0][0]

def information_gain(df,Y_ent):
	gain = []
	features = df.keys()[1:]
	for feature in features:
		#	Entropy of other features and thier values:

		print('\nFeature: ',feature)
		final_ent = entropy(df,feature,df['Y'])
		print('Gain of ',feature,'is: ',Y_ent - final_ent)
		gain.append(Y_ent - final_ent)
	print('\n**************************************\ngain: ',gain)
	return df.keys()[1:][np.argmax(gain)]

def decison_tree(df,Tree = None):
	#	Entropy of Output class
	y_unique, count_target = np.unique(df['Y'], return_counts=True)
	totalcount = np.sum(count_target)
	prob_target = count_target / totalcount
	Y_ent = 0
	for p in prob_target:
		if p != 0:
			Y_ent += -p * np.log2(p)
	print('************************************')
	print('\nEntropy of Y: ',Y_ent)

	#	function information gain return the node to be selected
	feature_node = information_gain(df, Y_ent)
	print('feature_node: ', feature_node)
	if Tree == None:
		Tree = {}
		Tree[feature_node] = {}
	feature_values,values_count = np.unique(df[feature_node], return_counts=True)
	print('feature_values: ',feature_values)
	print('values_count: ',values_count)
	for value in feature_values:
		#	Considering only those rows of all data whose value = feature node value
		new_df = df[df[feature_node]==value].reset_index(drop=True)
		# print(new_df)
		Y_unique,Y_count = np.unique(new_df['Y'],return_counts=True)
		print('Y_unique,Y_count: ',Y_unique,Y_count)
		#	stopping when only pure... i.e only e or p from Mushroom data remains
		if len(Y_count) == 1:
			Tree[feature_node][value] = Y_unique[0]
		else:
			# print('\n***************************************************\n')
			# pprint.pprint(Tree)
			# print('\n***************************************************\n')
			Tree[feature_node][value] = decison_tree(new_df)
	return Tree

def predict(tree,dfrow):
	for node,value in tree.items():
		tree = tree[node][dfrow[node]]
		if tree in ['e','p']:
			return tree

if __name__ == '__main__':
	df_train = pd.read_csv('MushroomTrain.csv', names=['Y', 'c_shape', 'c_surface', 'c_color', 'bruises','odor'],usecols=[0,1,2,3,4,5])
	df_test = pd.read_csv('MushroomTest.csv', names=['Y', 'c_shape', 'c_surface', 'c_color', 'bruises','odor'],usecols=[0,1,2,3,4,5])

	tree = decison_tree(df_train,None)
	print(tree)
	#	prediction ono training data
	train_predict = []
	for index, row in df_train.iterrows():
		train_predict.append(predict(tree,row))
	train_predict = np.array(train_predict)
	train_check = np.equal(train_predict,df_train['Y'])
	train_check_values,train_check_count = np.unique(train_check,return_counts=True)
	if len(train_check_values) == 1:
		if train_check_values == True:print('\nTrain accuracy: ',100,'%')
		if train_check_values == False: print('\nTrain accuracy: ', 0, '%')
	else:
		false_index = train_check.index('False')
		print('\nTrain accuracy: ',100*train_check_count[false_index]/np.sum[train_check_count])
	#	prediction ono testing data
	test_predict = []
	for index, row in df_test.iterrows():
		test_predict.append(predict(tree, row))
	test_predict = np.array(test_predict)
	test_check = np.equal(test_predict, df_test['Y'])
	test_check_values,test_check_count = np.unique(test_check,return_counts=True)
	if len(train_check_values) == 1:
		if test_check_values == True:print('\nTest accuracy: ',100,'%')
		if test_check_values == False: print('\nTest accuracy: ', 0, '%')
	else:
		false_index = test_check.index(False)
		print('\nTest accuracy: ', 100 * test_check_count[false_index] / np.sum[test_check_count])