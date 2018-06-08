import os
import jsonlines
import nltk
import codecs
import utilities

def getRelevantDocs(claim):
	docs = []
	entities = utilities.getNamedEntities(claim)
	for i in range(len(entities)):
		entities[i] = entities[i].replace(" ","_")
	docs = entities
	return docs