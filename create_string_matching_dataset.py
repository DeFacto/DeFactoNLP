import jsonlines
import doc_retrieval
import sentence_retrieval
import rte.rte as rte
import utilities
import spacy
import os

nlp = spacy.load('en_core_web_lg')
train_file = "data/train.jsonl"

train_file = jsonlines.open(train_file)
train_set = []

for lines in train_file:
	train_set.append(lines)

for example in train_set[:5]:
	entities = list(nlp(example['claim']).ents)
	for i in range(len(entities)):
		entities[i] = str(entities[i])
	print(example['claim'])
	print(entities)
	if example['verifiable'] == "VERIFIABLE":
		actual_evidence = example['evidence']
		actual_evidence_docs = []
		for i in range(len(actual_evidence_docs)):
			actual_evidence_docs.append(actual_evidence[2])
		print(actual_evidence_docs)