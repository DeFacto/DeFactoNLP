import os
import jsonlines
import codecs
import json
import numpy as np
import pickle
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import proof_extraction_train
from defacto.model_nl import ModelNL
from multiprocessing import Pool

entailment_predictions = "rte/entailment_predictions_test"
max_num_claims = 19998

def weed_out_evidence(claim_id):
	entailment_results_file = entailment_predictions + "/claim_" + str(claim_id) + ".json"
	entailment_results_file = codecs.open(entailment_results_file,"r","utf-8").readlines()
	weeded_out_file = codecs.open(entailment_predictions + "/corrected_" + str(claim_id) + ".json", "w+", "utf-8")
	try:
		defactoModel = None
		evidence_so_far = []
		for line in entailment_results_file:
			oriline = line
			line = json.loads(line)
			if defactoModel is None:
				defactoModel = ModelNL(claim=line['claim'])
			defacto_class=1
			evi = [line['premise_source_doc_id'],line['premise_source_doc_line_num']]
			if evi in evidence_so_far:
				continue
			else:
				evidence_so_far.append(evi)
			try:
				x = proof_extraction_train._extract_features(line['premise_source_doc_sentence'],line['claim'],defactoModel.triples)
				x = np.asarray(x)
				x = x.reshape(1, -1)
				defacto_clf = joblib.load('defacto/defacto_models/rfc.mod')
				y = defacto_clf.predict(x)
				defacto_class = y[0]
			except Exception as e:
				print("Bleh")
				raise e
			print(line['claim'])
			print(line['premise_source_doc_sentence'])
			print(defacto_class)
			if defacto_class == 0:
				continue
			else:
				weeded_out_file.write(oriline)
	except Exception as e:
		raise e
		for line in entailment_results_file:
			weeded_out_file.write(line)

if __name__ == '__main__':
	pool = Pool(processes=16)
	pool.map(weed_out_evidence,list(range(1,max_num_claims+1)))