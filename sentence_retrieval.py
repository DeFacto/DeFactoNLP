import os
import json
import jsonlines
import nltk
import codecs
import utilities

def getRelevantSentences(relevant_docs, entities, wiki_split_docs_dir):
	split_entities = []
	relevant_sentences = []
	for ent in entities:
		split_entities = split_entities + ent.split(" ")
	for relevant_doc in relevant_docs:
		file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".json","r","utf-8")
		file = json.load(file)
		full_lines = file["lines"]
		lines = []
		for line in full_lines:
			lines.append(line['content'])
		# file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
		# lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i].strip()
			lines[i] = lines[i].replace("-LRB-"," ( ")
			lines[i] = lines[i].replace("-RRB-"," ) ")
			if lines[i] == "":
				continue
			# split_line = lines[i].split(" ")
			# intersection = utilities.listIntersection(split_line,split_entities)
			# if len(intersection) > 0:
			temp = {}
			temp['id'] = relevant_doc
			temp['line_num'] = i
			temp['sentence'] = lines[i]
			relevant_sentences.append(temp)
	return relevant_sentences


def getSentence(wiki_doc_dir, doc_filename, sentence_id):
	
	doc = codecs.open(wiki_doc_dir + "/" + doc_filename + ".txt","r","utf-8")
	
	doc_splitted_lines= doc["lines"].split("\n")
	
	# assuming that the sentence id corresponds to the order of the sentences in the doc
	# sentences are organized as follows:
	#	SENTENCE_ID\tSENTENCE_TEXT\tNAMED_ENTITY1\tNAMED_ENTITY2
	return doc_splitted_lines[sentence_id].split("\t")[1]
