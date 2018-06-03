import os
import jsonlines
import nltk

def getRelevantDocs(claim):
	docs = []
	print claim
	# print nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(claim)))
	temp = open("temp.txt","w+")
	temp.write(claim)
	temp.close()
	cmd = "nohup java -mx600m -cp ner/stanford-ner/stanford-ner.jar:ner/stanford-ner/lib/* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile temp.txt > temp.tsv"
	os.system(cmd)
	temp = open("temp.tsv","r").readlines()
	entities = []
	for line in temp:
		line = line.strip().split("\t")
		if len(line)>1:
			ent = line[0]
			ent = ent.replace(" ","_")
			entities.append(ent)
	print entities
	return docs