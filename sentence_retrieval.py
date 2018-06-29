import os
import jsonlines
import nltk
import codecs
import utilities

def getRelevantSentences(relevant_docs, wiki_split_docs_dir):
	entities = []
	split_entities = []
	relevant_sentences = []
	for relevant_doc in relevant_docs:
		relevant_doc = relevant_doc.replace("-SLH-","/")
		relevant_doc = relevant_doc.replace("_"," ")
		entities.append(relevant_doc)
		split_entities = split_entities + relevant_doc.split(" ")
	for relevant_doc in relevant_docs:
		if not os.path.isfile(wiki_split_docs_dir + "/" + relevant_doc + ".txt"):
			continue
		file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i].strip()
			split_line = lines[i].split(" ")
			intersection = utilities.listIntersection(split_line,split_entities)
			if len(intersection) > 0:
				temp = {}
				temp['id'] = relevant_doc
				temp['line_num'] = i+1
				temp['sentence'] = lines[i]
				relevant_sentences.append(temp)
	return relevant_sentences