import os
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
		file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
		lines = file.readlines()
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
			temp['line_num'] = i+1
			temp['sentence'] = lines[i]
			relevant_sentences.append(temp)
	return relevant_sentences