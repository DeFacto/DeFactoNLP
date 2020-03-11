import jsonlines
import json
import doc_retrieval
import sentence_retrieval
import rte.rte as rte
import utilities
import spacy
import os
import codecs
import unicodedata as ud

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

relevant_sentences_file = "data/relevant_sentences_dev.jsonl"
instances = []

relevant_sentences_file = jsonlines.open(relevant_sentences_file)
model = "rte/fever_output/model.tar.gz"
model = load_archive(model)
predictor = Predictor.from_archive(model)

claim_num = 1

for line in relevant_sentences_file:
	instances.append(line)

def createTestSet(claim, candidateEvidences, claim_num):   
	testset = []
	for elem in candidateEvidences:
		testset.append({"hypothesis": claim, "premise": elem})
	return testset

def run_rte(claim,evidence,claim_num):
	fname = "claim_" + str(claim_num) + ".json"
	testset = createTestSet(claim,evidence,claim_num)
	preds = predictor.predict_batch_json(testset)
	return preds

for i in range(len(instances)):
	claim = instances[i]['claim']
	print(claim)
	evidence = instances[i]['predicted_sentences']
	potential_evidence_sentences = []
	for sentence in evidence:
		potential_evidence_sentences.append(sentence['sentence'])
	preds = run_rte(claim,potential_evidence_sentences,claim_num)

	saveFile = codecs.open("rte/entailment_predictions/claim_" + str(claim_num) + ".json", mode = "w+", encoding="utf-8")
	for i in range(len(preds)):
		preds[i]['claim'] = claim
		preds[i]['premise_source_doc_id'] = evidence[i]['id']
		preds[i]['premise_source_doc_line_num'] = evidence[i]['line_num']
		preds[i]['premise_source_doc_sentence'] = evidence[i]['sentence']
		saveFile.write(json.dumps(preds[i],ensure_ascii=False) + "\n")

	saveFile.close()
	claim_num += 1