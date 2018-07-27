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
		support_max_conf_score= 0.0
		refute_max_conf_score= 0.0
		nei_max_conf_score= 0.0
		
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
			
			
			if nei_max_conf_score < line["label_probs"][0]:
				nei_max_conf_score= line["label_probs"][0]
			
			if support_max_conf_score < line["label_probs"][1]:
				support_max_conf_score= line["label_probs"][1]
			
			if refute_max_conf_score < line["label_probs"][2]:
				refute_max_conf_score= line["label_probs"][2]

		features = [nei_max_conf_score, support_max_conf_score, refute_max_conf_score,
					nei_count,nei_confidence,support_count,support_confidence,refute_count,refute_confidence]

		if nei_count!=0:
			features.append(float(nei_confidence) / float(nei_count))
		else:
			features.append(0)
		if support_count!=0:
			features.append(float(support_confidence) / float(support_count))
		else:
			features.append(0)
		if refute_count!=0:
			features.append(float(refute_confidence) / float(refute_count))
		else:
			features.append(0)
			
		x_train.append(features)
		
		#TODO:
		# get defacto features here, if required
		
		i += 1
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
			support_scores = []
			refute_scores = []
			nei_scores = []
			support_max_conf_score= 0.0
			refute_max_conf_score= 0.0
			nei_max_conf_score= 0.0
			for line in entailment_results_file:
				line = json.loads(line)
				evi = [line['premise_source_doc_id'],line['premise_source_doc_line_num']]
				maxIndex= np.argmax(np.asarray(line["label_probs"]))
				if maxIndex == 0:
					nei_count += 1
					nei_confidence += line["label_probs"][maxIndex]					
					nei_evidence.append(evi)
					nei_scores.append(line["label_probs"][0])
				elif maxIndex == 1:
					support_count += 1
					support_confidence += line["label_probs"][maxIndex]
					support_evidence.append(evi)
					support_scores.append(line["label_probs"][1])
				else:
					refute_count += 1
					refute_confidence += line["label_probs"][maxIndex]
					refute_evidence.append(evi)
					refute_scores.append(line["label_probs"][2])
				if nei_max_conf_score < line["label_probs"][0]:
					nei_max_conf_score= line["label_probs"][0]			
				if support_max_conf_score < line["label_probs"][1]:
					support_max_conf_score= line["label_probs"][1]
				if refute_max_conf_score < line["label_probs"][2]:
					refute_max_conf_score= line["label_probs"][2]
			# print(support_scores)
			# print(support_evidence)
			# print(support_count)	
					
			features = [nei_max_conf_score, support_max_conf_score, refute_max_conf_score,
						nei_count,nei_confidence,support_count,support_confidence,refute_count,refute_confidence]

			if nei_count!=0:
				features.append(float(nei_confidence) / float(nei_count))
				nei_scores, nei_evidence = zip(*sorted(zip(nei_scores, nei_evidence), reverse=True))
			else:
				features.append(0)
			if support_count!=0:
				features.append(float(support_confidence) / float(support_count))
				support_scores, support_evidence = zip(*sorted(zip(support_scores, support_evidence), reverse=True))
			else:
				features.append(0)
			if refute_count!=0:
				features.append(float(refute_confidence) / float(refute_count))
				refute_scores, refute_evidence = zip(*sorted(zip(refute_scores, refute_evidence), reverse=True))
			else:
				features.append(0)
			# get defacto features here, if required
			features = np.asarray(features)
			features = features.reshape(1,features.shape[0])
			# print(features)
			new_pred['predicted_label'] = intolabel[clf.predict(features)[0]]
			if new_pred['predicted_label'] == "SUPPORTS":
				if support_count == 0:
					new_pred['predicted_label'] = "NOT ENOUGH INFO"
					new_pred['predicted_evidence'] = []
				else:
					new_pred['predicted_evidence'] = support_evidence[:5]
			elif new_pred['predicted_label'] == "REFUTES":
				if refute_count == 0:
					new_pred['predicted_label'] = "NOT ENOUGH INFO"
					new_pred['predicted_evidence'] = []
				else:
					new_pred['predicted_evidence'] = refute_evidence[:5]
			else:
				new_pred['predicted_evidence'] = []
			writer.write(new_pred)
			i += 1

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
