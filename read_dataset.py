import jsonlines
import doc_retrieval
import sentence_retrieval
import rte.rte as rte
import utilities
import spacy
import os
# from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_lg')

train_file = "data/train.jsonl"
test_file = "data/shared_task_dev_public.jsonl"
dev_file = "data/dev.jsonl"
results_file = "predictions/predictions.jsonl"

wiki_dir = 'data/wiki-pages/wiki-pages'
wiki_split_docs_dir = "data/wiki-pages/wiki-pages-split"

train_file = jsonlines.open(train_file)
test_file = jsonlines.open(test_file)
dev_file = jsonlines.open(dev_file)

train_set = []
test_set = []
dev_set = []

wiki_entities = os.listdir(wiki_split_docs_dir)
for i in range(len(wiki_entities)):
	wiki_entities[i] = wiki_entities[i].replace("-SLH-","/")
	wiki_entities[i] = wiki_entities[i].replace("_"," ")
	wiki_entities[i] = wiki_entities[i][:-4]

# for lines in train_file:
# 	train_set.append(lines)
for lines in test_file:
	test_set.append(lines)
# for lines in dev_file:
# 	dev_set.append(lines)

total = 0.0
retrieved = 0.0

# # adding entities to gazetter
# matcher = Matcher(nlp.vocab,100)
# gazetter = "list_of_Wikipedia_files.txt"
# gazetter = open(gazetter,"r").readlines()
# for i in range(len(gazetter)):
# 	gazetter[i] = gazetter[i][:-5]
# 	gazetter[i] = gazetter[i].replace("-SLH-","/")
# 	gazetter[i] = gazetter[i].replace("_"," ")
# 	matcher.add(gazetter[i],utilities.on_match,nlp(gazetter[i]))

with jsonlines.open(results_file, mode='w') as writer:
	for example in test_set[:5]:
		relevant_docs,entities = doc_retrieval.getRelevantDocs(example['claim'],wiki_entities,"spaCy",nlp)
		print(example['claim'])
		print(entities)
		print(relevant_docs)
		relevant_sentences = sentence_retrieval.getRelevantSentences(relevant_docs,entities,wiki_split_docs_dir)
		print(relevant_sentences)
		result = rte.textual_entailment_evidence_retriever(example['claim'],relevant_sentences)
		print(result)
		final_result = {}
		final_result['id'] = example['id']
		final_result['predicted_label'] = result['label']
		predicted_evidence = []
		for evidence in result['evidence']:
			predicted_evidence.append([evidence['id'],evidence['line_num']])
		final_result['predicted_evidence'] = predicted_evidence
		writer.write(final_result)
	# 	if example['label']=="REFUTES" or example['label']=="SUPPORTS":
	# 		for actual_evidence in example['evidence'][0]:
	# 			total+=1.0
	# 			if actual_evidence[2] in relevant_docs:
	# 				retrieved += 1.0
	# 				# print actual_evidence[2]

# print total
# print retrieved
# print retrieved/total