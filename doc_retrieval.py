import os
import jsonlines
import nltk
import codecs
import utilities
import spacy
import stringdist
from spacy.matcher import PhraseMatcher
import unicodedata as ud

def getClosestDocs(wiki_entities,entities):
	entities = list(entities)
	for i in range(len(entities)):
		entities[i] = str(entities[i])
	selected_docs = []
	for ent in entities:
		ent = ud.normalize('NFC',ent)
		if ent in wiki_entities:
			best_match = ent
		else:
			best = 1.1
			best_match = ""
			for we in wiki_entities:
				dist = stringdist.levenshtein_norm(we,ent)
				if dist < best:
					best = dist
					best_match = we
		best_match = best_match.replace(" ","_")
		best_match = best_match.replace("/","-SLH-")
		best_match = best_match.replace("(","-LRB-")
		best_match = best_match.replace(")","-RRB-")
		selected_docs.append(best_match)
	return selected_docs, entities

def getRelevantDocs(claim,wiki_entities,ner_module="spaCy",nlp=None): #,matcher=None,nlp=None
	entities = []
	if ner_module=='spaCy' and nlp is not None: # and matcher is not None 
		# entities = utilities.getNamedEntitiesspaCy(claim,matcher,nlp)
		entities = list(nlp(claim).ents)
	elif ner_module=='StanfordNER':
		entities = utilities.getNamedEntitiesStanfordNER(claim)
	else:
		print("Error: Incorrect Document Retrieval Specifications")
		return
	return getClosestDocs(wiki_entities,entities)