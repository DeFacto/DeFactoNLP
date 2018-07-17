import os
import jsonlines
import nltk
import codecs
from spacy.matcher import PhraseMatcher

def getNamedEntitiesStanfordNER(sentence):
	# print sentence
	# print nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
	temp = codecs.open("temp.txt","w+","utf-8")
	temp.write(sentence)
	temp.close()
	cmd = "nohup java -mx600m -cp ner/stanford-ner/stanford-ner.jar:ner/stanford-ner/lib/* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile temp.txt > temp.tsv"
	os.system(cmd)
	temp = codecs.open("temp.tsv","r","utf-8").readlines()
	entities = []
	for line in temp:
		line = line.strip().split("\t")
		if len(line)>1:
			ent = line[0]
			entities.append(ent)
	return entities

def listIntersection(list1,list2):
	s = set(list2)
	return [val for val in list1 if val in s]
