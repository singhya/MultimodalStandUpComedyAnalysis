__credits__ = ["Harsh Fatepuria","Rahul Agrawal"]
__email__ = "fatepuri@usc.edu, rahulagr@usc.edu"

'''
command to execute: 

	python CS535MLClassifier.py -svm -svm3 -svmrbf -nb -nn

note: needs sklearn, statistics and numpy python libraries
'''

import statistics
import math
import sys
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


class CS535MLClassifier:
	def __init__(self,filename):
		self.dataset_file_name=filename

	def get_fold_test_and_train_only(self,typeOfTest,test_start,test_end):
		input_file=open(self.dataset_file_name , 'r')
		dataset=input_file.read().split("\n")

		if typeOfTest==1:
			# accoustic
			start_column=3
			end_column=5
		elif typeOfTest==0:
			# visual
			start_column=6
			end_column=8
		else:
			# multimodal
			start_column=3
			end_column=8

		test1_data=list()
		test1_label=list()
		train1_data=list()
		train1_label=list()

		for i in range(test_start,test_end+1):
			row=list()
			data=dataset[i].split(",")
			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			test1_data.append(row)
			test1_label.append(int(data[2]))

		for i in range(1,test_start):
			row=list()
			data=dataset[i].split(",")
			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			train1_data.append(row)
			train1_label.append(int(data[2]))

		for i in range(test_end+1,281):
			row=list()
			data=dataset[i].split(",")
			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			train1_data.append(row)
			train1_label.append(int(data[2]))

		return test1_data,test1_label,train1_data,train1_label


	def get_fold(self,typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end):
		input_file=open(self.dataset_file_name , 'r')
		dataset=input_file.read().splitlines()

		if typeOfTest==1:
			# accoustic
			start_column=3
			end_column=5
		elif typeOfTest==0:
			# visual
			start_column=6
			end_column=8
		else:
			# multimodal
			start_column=3
			end_column=8

		test1_data=list()
		test1_label=list()
		validate1_data=list()
		validate1_label=list()
		train1_data=list()
		train1_label=list()

		for i in range(test_start,test_end+1):
			row=list()
			data=dataset[i].split(",")

			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			test1_data.append(row)
			test1_label.append(int(data[2]))

		for i in range(validate_start,validate_end+1):
			row=list()
			data=dataset[i].split(",")
			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			validate1_data.append(row)
			validate1_label.append(int(data[2]))

		for i in range(train_start,train_end+1):
			row=list()
			data=dataset[i].split(",")
			for j in range(start_column,end_column+1):
				row.append(float(data[j]))
			train1_data.append(row)
			train1_label.append(int(data[2]))
		return test1_data,test1_label,validate1_data,validate1_label,train1_data,train1_label


	def perform_naivebayes(self,typeOfTest,test_start,test_end):
		test1_data,test1_label,train1_data,train1_label = self.get_fold_test_and_train_only(typeOfTest,test_start,test_end)
		clf = GaussianNB()
		test_results=list()
		clf.fit(train1_data, train1_label)
		for line in test1_data:
			predicted = clf.predict([line])[0]
			test_results.append(predicted)
		print "Accuracy in test: ",metrics.accuracy_score(test1_label,test_results)


	def perform_svm(self,typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end):
		test1_data,test1_label,validate1_data,validate1_label,train1_data,train1_label = self.get_fold(typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end)
		clf = svm.LinearSVC()
		accuracy=list()

		for c in [0.001,0.01,0.1,1,10,100]:
			validation_result=list()
			clf.set_params(C=c)
			clf.fit(train1_data, train1_label)
			correct_classification = 0
			i=0
			for line in validate1_data:
				validation_result.append(clf.predict([line])[0])
			accuracy.append(metrics.accuracy_score(validate1_label,validation_result))
		print "Accuracy in validation (max): ",max(accuracy) 

		cMax=math.pow(10,(accuracy.index(max(accuracy))-3))
		print "Hyperparameter(c) for max accuracy = ",cMax

		clf = svm.LinearSVC()
		test_results=list()
		train_results=list()
		clf.set_params(C=cMax)
		clf.fit(train1_data, train1_label)
		for line in train1_data:
			train_results.append(clf.predict([line])[0])
		print "Accuracy in training: ",metrics.accuracy_score(train1_label,train_results)
		
		for line in test1_data:
			predicted = clf.predict([line])[0]
			test_results.append(predicted)
		print "\nAccuracy in test: ",metrics.accuracy_score(test1_label,test_results)


	def perform_svm_n_fold(self,num_folds,typeOfTest,test_start,test_end):
		test1_data,test1_label,train1_data,train1_label = self.get_fold_test_and_train_only(typeOfTest,test_start,test_end)
		k_fold = KFold(n_splits=num_folds)
		accuracy=list()
		for c in [0.001,0.01,0.1,1,10,100]:
			clf = svm.LinearSVC(C=c)
			accuracy.append(statistics.mean(cross_val_score(clf,train1_data,train1_label,cv=k_fold,n_jobs=-1)))
		print "Accuracy in validation(max): ",max(accuracy)
		print "Accuracy in validation(average): ",statistics.mean(accuracy)
		cMax=math.pow(10,(accuracy.index(max(accuracy))-3))
		print "Hyperparameter(c) for max accuracy = ",cMax

		clf = svm.LinearSVC()
		test_results=list()
		train_results=list()
		clf.set_params(C=cMax)
		clf.fit(train1_data, train1_label)

		for line in train1_data:
			train_results.append(clf.predict([line])[0])
		print "Accuracy in training: ",metrics.accuracy_score(train1_label,train_results)

		for line in test1_data:
			test_results.append(clf.predict([line])[0])
		print "Accuracy in test: ",metrics.accuracy_score(test1_label,test_results)


	def perform_svm_rbf(self,typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end):
		test1_data,test1_label,validate1_data,validate1_label,train1_data,train1_label = self.get_fold(typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end)
		clf = svm.SVC()
		accuracy=list()
		for c in [0.001,0.01,0.1,1,10,100]:
			validation_result=list()
			clf.set_params(C=c,kernel="rbf")
			clf.fit(train1_data, train1_label)
			correct_classification = 0
			i=0
			for line in validate1_data:
				validation_result.append(clf.predict([line])[0])
			accuracy.append(metrics.accuracy_score(validate1_label,validation_result))

		cMax=math.pow(10,(accuracy.index(max(accuracy))-3))
		print "Hyperparameter(c) for max accuracy = ",cMax

		clf = svm.SVC(C=cMax,kernel="rbf")
		test_results=list()
		clf.fit(train1_data, train1_label)
		for line in test1_data:
			test_results.append(clf.predict([line])[0])
		print "Accuracy in test: ",metrics.accuracy_score(test1_label,test_results)


	def perform_nn(self,typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end):		
		test1_data,test1_label,validate1_data,validate1_label,train1_data,train1_label = self.get_fold(typeOfTest,test_start,test_end,validate_start,validate_end,train_start,train_end)
		accuracy=list()
		hidden_layers=list()
		for ii in range(1,11):
			hidden_layers.append((ii,))
		for hidden_layer in hidden_layers:
			clf = MLPClassifier(solver = 'lbfgs',alpha = 1, hidden_layer_sizes = hidden_layer, random_state = 1)
			clf.fit(train1_data, train1_label)
			
			correct_classification=0
			i=0
			for line in validate1_data:
				if clf.predict([line])[0] == validate1_label[i] :
					correct_classification = correct_classification + 1
				i=i+1
			accuracy.append(correct_classification/float(len(validate1_label)))

		hidden_layer_for_highest_accuracy=hidden_layers[accuracy.index(max(accuracy))]
		print "Hyperparameter(c) for max accuracy = ",hidden_layer_for_highest_accuracy

		clf = MLPClassifier(solver = 'lbfgs',alpha = 1, hidden_layer_sizes = hidden_layer_for_highest_accuracy, random_state = 1)
		clf.fit(train1_data, train1_label)

		test_results=list()
		for line in test1_data:
			test_results.append(clf.predict([line])[0])
		print "Accuracy in test: ",metrics.accuracy_score(test1_label,test_results)



def naive_bayes(a):
	print "\n\n\nNAIVE BAYES"
	print "-------------------"
	print "\nExperiment 1: "
	print "-----------------"
	test_start=1
	test_end=67
	print "Multimodal:"
	a.perform_naivebayes(2,test_start,test_end)
	print "\nAudio:"
	a.perform_naivebayes(1,test_start,test_end)
	print "\nVisual:"
	a.perform_naivebayes(0,test_start,test_end)


	print "\n\nExperiment 2: "
	print "-----------------"
	test_start=212
	test_end=280
	print "Multimodal:"
	a.perform_naivebayes(2,test_start,test_end)
	print "\nAudio:"
	a.perform_naivebayes(1,test_start,test_end)
	print "\nVisual:"
	a.perform_naivebayes(0,test_start,test_end)


	print "\n\nExperiment 3: "
	print "-----------------"
	test_start=141
	test_end=211
	print "Multimodal:"
	a.perform_naivebayes(2,test_start,test_end)
	print "\nAudio:"
	a.perform_naivebayes(1,test_start,test_end)
	print "\nVisual:"
	a.perform_naivebayes(0,test_start,test_end)


	print "\n\nExperiment 4: "
	print "-----------------"
	test_start=68
	test_end=140
	print "Multimodal:"
	a.perform_naivebayes(2,test_start,test_end)
	print "\nAudio:"
	a.perform_naivebayes(1,test_start,test_end)
	print "\nVisual:"
	a.perform_naivebayes(0,test_start,test_end)


def neural_net(a):
	print "\n\n\nNEURAL NETWORK"
	print "------------------"
	print "\nExperiment 1: "
	print "-----------------"
	test_start=1
	test_end=67
	validate_start=68
	validate_end=140
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_nn(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_nn(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_nn(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 2: "
	print "-----------------"
	test_start=212
	test_end=280
	validate_start=141
	validate_end=211
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_nn(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_nn(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_nn(0,test_start,test_end,validate_start,validate_end,train_start,train_end)

	print "\n\nExperiment 3: "
	print "-----------------"
	test_start=141
	test_end=211
	validate_start=212
	validate_end=280
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_nn(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_nn(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_nn(0,test_start,test_end,validate_start,validate_end,train_start,train_end)

	print "\n\nExperiment 4: "
	print "-----------------"
	test_start=68
	test_end=140
	validate_start=1
	validate_end=67
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_nn(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_nn(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_nn(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


def svm_n_folds(a,numFolds):
	print "\n\n\nSVM with ",numFolds," folds and not respecting speaker independence"
	print "---------------------------------------------------------------------------"
	print "\nExperiment 1: "
	print "-----------------"
	test_start=1
	test_end=67
	a.perform_svm_n_fold(numFolds,2,test_start,test_end)

	print "\n\nExperiment 2: "
	print "-----------------"
	test_start=212
	test_end=280
	a.perform_svm_n_fold(numFolds,2,test_start,test_end)

	print "\n\nExperiment 3: "
	print "-----------------"
	test_start=141
	test_end=211
	a.perform_svm_n_fold(numFolds,2,test_start,test_end)

	print "\n\nExperiment 4: "
	print "-----------------"
	test_start=68
	test_end=140
	a.perform_svm_n_fold(numFolds,2,test_start,test_end)


def svm_rbf(a):
	print "\n\n\nSVM with RBF Kernel"
	print "---------------------------"
	print "\nExperiment 1: "
	print "-----------------"
	test_start=1
	test_end=67
	validate_start=68
	validate_end=140
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_svm_rbf(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm_rbf(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm_rbf(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 2: "
	print "-----------------"
	test_start=212
	test_end=280
	validate_start=141
	validate_end=211
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_svm_rbf(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm_rbf(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm_rbf(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 3: "
	print "-----------------"
	test_start=141
	test_end=211
	validate_start=212
	validate_end=280
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_svm_rbf(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm_rbf(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm_rbf(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 4: "
	print "-----------------"
	test_start=68
	test_end=140
	validate_start=1
	validate_end=67
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_svm_rbf(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm_rbf(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm_rbf(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


def svm_linear(a):
	print "\n\n\nLinear SVM"
	print "--------------------"
	print "\nExperiment 1: "
	print "-----------------"
	test_start=1
	test_end=67
	validate_start=68
	validate_end=140
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_svm(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm(0,test_start,test_end,validate_start,validate_end,train_start,train_end)



	print "\n\nExperiment 2: "
	print "-----------------"
	test_start=212
	test_end=280
	validate_start=141
	validate_end=211
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_svm(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 3: "
	print "-----------------"
	test_start=141
	test_end=211
	validate_start=212
	validate_end=280
	train_start=1
	train_end=140

	print "Multimodal:"
	a.perform_svm(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


	print "\n\nExperiment 4: "
	print "-----------------"
	test_start=68
	test_end=140
	validate_start=1
	validate_end=67
	train_start=141
	train_end=280

	print "Multimodal:"
	a.perform_svm(2,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nAudio:"
	a.perform_svm(1,test_start,test_end,validate_start,validate_end,train_start,train_end)
	print "\nVisual:"
	a.perform_svm(0,test_start,test_end,validate_start,validate_end,train_start,train_end)


if len(sys.argv) == 1:
	print "Please specifiy the classifier(s) to execute...\n"
else:
	a=CS535MLClassifier('dataset.csv')
	args=sys.argv[1:]
	args=list(set(args))

	for arg in args:
		if arg=="-nb":
			naive_bayes(a)
		if arg=="-svm":
			svm_linear(a)
		if arg=="-svm3":
			svm_n_folds(a,3)
		if arg=="-svm4":
			svm_n_folds(a,4)
		if arg=="-nn":
			neural_net(a)
		if arg=="-svmrbf":
			svm_rbf(a)
