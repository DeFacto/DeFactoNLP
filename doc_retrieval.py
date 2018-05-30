import os
import jsonlines

def getRelevantDocs(claim):
	docs = []
	print claim
	temp = open("temp.txt","w+")
	temp.write(claim)
	temp.close()
	cmd = "java -mx600m -cp ner/stanford-ner/stanford-ner.jar:ner/stanford-ner/lib/* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile temp.txt > temp.tsv"
	os.system(cmd)
	temp = open("temp.tsv","r").readlines()
	entities = []
	for line in temp:
		line = line.strip().split("\t")
		if len(line)>1:
			entities.append(line[0])
	print entities
	return docs