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

def getDocContent(wiki_folder, doc_id):
	
	for currentFile in os.listdir(wiki_folder): 
		fileContent= jsonlines.open(wiki_folder + "/" + currentFile)
		
		for doc in fileContent:
			
			if doc["id"] == doc_id:
				# add file id where the doc was found. This can be useful for next steps of the process to get document content without requiring an exhaustive search on all the files.
				doc["fileId"] = currentFile
				return doc
			
	
	return None

def getDocContentFromFile(wiki_folder, doc_filename, doc_id):
	
	fileContent= jsonlines.open(wiki_folder + "/" + doc_filename)
	
	for doc in fileContent:
		
		if doc["id"] == doc_id:
			doc["fileId"] = doc_filename
			return doc
		
	return None

def preProcessDoc(doc):
	
	# process "lines"
	doc_splitted_lines= doc["lines"].split("\n")
	
	linesList= []
	
	for line in doc_splitted_lines:
		# sentences are organized as follows:
		#	SENTENCE_ID\tSENTENCE_TEXT\tNAMED_ENTITY1\tNAMED_ENTITY2
		splittedSentence= line.split("\t")
		if len(splittedSentence) >= 3:
			linesList.append({"content": splittedSentence[1], "namedEntitiesList": splittedSentence[2:]})
		else:
			linesList.append({"content": splittedSentence[1], "namedEntitiesList": []})
	
	return {"id": doc["id"], "fileId": doc["fileId"],"text": doc["text"], "lines": linesList}
