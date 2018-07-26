import os
import jsonlines
import codecs
import json
import numpy as np
import pickle
from sklearn import svm
from sklearn.externals import joblib

labeltoint = {}
labeltoint['NOT ENOUGH INFO'] = 0
labeltoint['SUPPORTS'] = 1
labeltoint['REFUTES'] = 2
intolabel = ['NOT ENOUGH INFO','SUPPORTS','REFUTES']

# generate features for train set
def populate_train(gold_train,entailment_predictions_train):
	x_train = []
	y_train = []
	i = 1
	gold_train = jsonlines.open(gold_train)
	for instance in gold_train:
		y_train.append(labeltoint[instance['label']])
		entailment_results_file = entailment_predictions_train + "/claim_" + str(i) + ".json"
		entailment_results_file = codecs.open(entailment_results_file,"r","utf-8").readlines()
		support_count = 0
		refute_count = 0
		nei_count = 0
		support_confidence = 0
		refute_confidence = 0
		nei_confidence = 0
		for line in entailment_results_file:
			line = json.loads(line)
			maxIndex= np.argmax(np.asarray(line["label_probs"]))
			if maxIndex == 0:
				nei_count += 1
				nei_confidence += line["label_probs"][maxIndex]
			elif maxIndex == 1:
				support_count += 1
				support_confidence += line["label_probs"][maxIndex]
			else:
				refute_count += 1
				refute_confidence += line["label_probs"][maxIndex]
		features = [nei_count,nei_confidence,support_count,support_confidence,refute_count,refute_confidence]
		x_train.append(features)
		# get defacto features here, if required
		i += 1
		if i == 8000:
			break
	return (x_train,y_train)

def predict_test(predictions_test,entailment_predictions_test,new_predictions_file):
	clf = joblib.load('label_classifier.pkl')
	i = 1
	previous_predictions = jsonlines.open(predictions_test)
	with jsonlines.open(new_predictions_file, mode='w') as writer:
		for pred in previous_predictions:
			new_pred = {}
			new_pred['id'] = pred['id']
			new_pred['predicted_evidence'] = []
			entailment_results_file = entailment_predictions_test + "/claim_" + str(i) + ".json"
			entailment_results_file = codecs.open(entailment_results_file,"r","utf-8").readlines()
			support_evidence = []
			refute_evidence = []
			nei_evidence = []
			support_count = 0
			refute_count = 0
			nei_count = 0
			support_confidence = 0
			refute_confidence = 0
			nei_confidence = 0
			for line in entailment_results_file:
				line = json.loads(line)
				evi = [line['premise_source_doc_id'],line['premise_source_doc_line_num']]
				maxIndex= np.argmax(np.asarray(line["label_probs"]))
				if maxIndex == 0:
					nei_count += 1
					nei_confidence += line["label_probs"][maxIndex]					
					nei_evidence.append(evi)
				elif maxIndex == 1:
					support_count += 1
					support_confidence += line["label_probs"][maxIndex]
					support_evidence.append(evi)
				else:
					refute_count += 1
					refute_confidence += line["label_probs"][maxIndex]
					refute_evidence.append(evi)
			features = [nei_count,nei_confidence,support_count,support_confidence,refute_count,refute_confidence]
			# get defacto features here, if required
			features = np.asarray(features)
			features = features.reshape(1,features.shape[0])
			new_pred['predicted_label'] = labeltoint[clf.predict(features)[0]]
			if new_pred['predicted_label'] == "SUPPORTS":
				new_pred['predicted_evidence'] = support_evidence
			elif new_pred['predicted_label'] == "REFUTES":
				new_pred['predicted_evidence'] = refute_evidence
			else:
				new_pred['predicted_evidence'] = nei_evidence
			writer.write(new_pred)

predictions_train = "predictions_train.jsonl"
predictions_test = "predictions.jsonl"
new_predictions_file = "new_predictions.jsonl"

gold_train = "data/subsample_train_relevant_docs.jsonl"
entailment_predictions_train = "rte/entailment_predictions_train"
entailment_predictions_test = "rte/entailment_predictions_test"

x_train, y_train = populate_train(gold_train,entailment_predictions_train)
# x_test = x_train[6000:]
# y_test = y_train[6000:]

# x_train = x_train[:6000]
# y_train = y_train[:6000]
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test)

print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)

clf = svm.SVC()
clf.fit(x_train,y_train)

joblib.dump(clf, 'label_classifier.pkl') 
# clf = joblib.load('filename.pkl')

# print(clf.score(x_test,y_test))

predict_test(predictions_test,entailment_predictions_test,new_predictions_file)