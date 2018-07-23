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

nlp = spacy.load('en_core_web_lg')

test_file = "data/shared_task_dev_public_relevant_docs.jsonl"
results_file = "predictions.jsonl"

wiki_dir = 'data/wiki-pages/wiki-pages'
wiki_split_docs_dir = "data/wiki-pages/wiki-pages-split"

test_file = jsonlines.open(test_file)
test_set = []

claim_id = 1

wiki_entities = os.listdir(wiki_split_docs_dir)
for i in range(len(wiki_entities)):
	wiki_entities[i] = wiki_entities[i].replace("-SLH-","/")
	wiki_entities[i] = wiki_entities[i].replace("_"," ")
	wiki_entities[i] = wiki_entities[i][:-4]
	wiki_entities[i] = wiki_entities[i].replace("-LRB-","(")
	wiki_entities[i] = wiki_entities[i].replace("-RRB-",")")

for lines in test_file:
	lines['claim'] = lines['claim'].replace("-LRB-"," ( ")
	lines['claim'] = lines['claim'].replace("-RRB-"," ) ")
	test_set.append(lines)

with jsonlines.open(results_file, mode='w') as writer:
	for example in test_set:
		relevant_docs,entities = doc_retrieval.getRelevantDocs(example['claim'],wiki_entities,"StanfordNER",nlp)
		relevant_docs = list(set(relevant_docs))
		print(example['claim'])
		relevant_sentences = sentence_retrieval.getRelevantSentences(relevant_docs,entities,wiki_split_docs_dir)
		for i in range(len(example['predicted_sentences'])):
			relevant_doc = ud.normalize('NFC',example['predicted_sentences'][i][0])
			relevant_doc = relevant_doc.replace("/","-SLH-")
			file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".json","r","utf-8")
			file = json.load(file)
			full_lines = file["lines"]
			lines = []
			for line in full_lines:
				lines.append(line['content'])
			# file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
			# lines = file.readlines()
			lines[example['predicted_sentences'][i][1]-1] = lines[example['predicted_sentences'][i][1]-1].strip()
			lines[example['predicted_sentences'][i][1]-1] = lines[example['predicted_sentences'][i][1]-1].replace("-LRB-"," ( ")
			lines[example['predicted_sentences'][i][1]-1] = lines[example['predicted_sentences'][i][1]-1].replace("-RRB-"," ) ")
			if lines[example['predicted_sentences'][i][1]-1] == "":
				continue
			temp = {}
			temp['id'] = relevant_doc
			temp['line_num'] = example['predicted_sentences'][i][1]
			temp['sentence'] = lines[example['predicted_sentences'][i][1]-1]
			relevant_sentences.append(temp)
		# print(relevant_sentences)
		relevant_docs = relevant_docs + list(example['predicted_pages'])
		relevant_docs = list(set(relevant_docs))
		result = rte.textual_entailment_evidence_retriever(example['claim'],relevant_sentences,claim_id)
		claim_id = claim_id + 1
		final_result = {}
		final_result['id'] = example['id']
		final_result['predicted_label'] = result['label']
		predicted_evidence = []
		for evidence in result['evidence']:
			if [evidence['id'],evidence['line_num']] not in predicted_evidence:
				predicted_evidence.append([evidence['id'],evidence['line_num']])
		final_result['predicted_evidence'] = predicted_evidence
		writer.write(final_result)