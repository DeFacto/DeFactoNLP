import os
import jsonlines
import json
import codecs
import utilities
import stringdist
import unicodedata as ud
import clausiepy.clausiepy as clausie


def clean_entities(entities):
    entities = list(entities)
    ents_to_remove = set()
    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j:
                continue
            if entities[i] in entities[j]:
                # keep the smaller ones...
                ents_to_remove.add(entities[j])
                # or keep the bigger one...
                # ents_to_remove.add(entities[i])
    for ent in ents_to_remove:
        entities.remove(ent)

    return entities


def get_docs_with_oie(claim, wiki_entities,client):
    ents = set()

    # triple extraction standfordIE
    triples = client.annotate(claim)
    for triple in triples:
        ents.add(triple["subject"])
        ents.add(triple["object"])

    # triples extraction clausIE
    if len(triples) == 0:
        clauses = clausie.clausie(claim)
        for clause in clauses:
            for sub in clause['S']:
                ents.add(sub)
            for obj in clause['O']:
                ents.add(obj)

    if len(ents) > 4:
        ents = clean_entities(ents)
    docs, entities = getClosestDocs(wiki_entities, ents)

    return docs, entities


def getClosestDocs(wiki_entities, entities):
    entities = list(entities)
    for i in range(len(entities)):
        entities[i] = str(entities[i])
    selected_docs = []
    for ent in entities:
        ent = ud.normalize('NFC', ent)
        if ent in wiki_entities:
            best_match = ent
        else:
            best = 1.1
            best_match = ""
            for we in wiki_entities:
                dist = stringdist.levenshtein_norm(we, ent)
                if dist < best:
                    best = dist
                    best_match = we
        best_match = best_match.replace(" ", "_")
        best_match = best_match.replace("/", "-SLH-")
        best_match = best_match.replace("(", "-LRB-")
        best_match = best_match.replace(")", "-RRB-")
        selected_docs.append(best_match)
    return selected_docs, entities


def getRelevantDocs(claim, wiki_entities, ner_module="spaCy", nlp=None):  # ,matcher=None,nlp=None
    entities = []
    if ner_module == 'spaCy' and nlp is not None:  # and matcher is not None
        # entities = utilities.getNamedEntitiesspaCy(claim,matcher,nlp)
        entities = list(nlp(claim).ents)
    elif ner_module == 'StanfordNER':
        entities = utilities.getNamedEntitiesStanfordNER(claim)
    else:
        print("Error: Incorrect Document Retrieval Specifications")
        return
    return getClosestDocs(wiki_entities, entities)


def getDocContent(wiki_folder, doc_id):
    for currentFile in os.listdir(wiki_folder):
        fileContent = jsonlines.open(wiki_folder + "/" + currentFile)

        for doc in fileContent:

            if doc["id"] == doc_id:
                # add file id where the doc was found. This can be useful for next steps of the process to get document content without requiring an exhaustive search on all the files.
                doc["fileId"] = currentFile
                return doc

    return None


"""
def getDocContentFromFile(wiki_folder, doc_filename, doc_id):
	
	fileContent= jsonlines.open(wiki_folder + "/" + doc_filename)
	
	for doc in fileContent:
		
		if doc["id"] == doc_id:
			doc["fileId"] = doc_filename
			return doc
		
	return None
"""


def getDocContentFromFile(wiki_folder, doc_filename):
    try:
        file = codecs.open(wiki_folder + "/" + doc_filename + ".json")
        fileContent = json.load(file)
        return fileContent
    except:
        print("Could not find or open file: ")
        print(doc_filename)
        print("")
        return None


def preProcessDoc(doc):
    # process "lines"
    doc_splitted_lines = doc["lines"].split("\n")

    linesList = []

    for line in doc_splitted_lines:
        # sentences are organized as follows:
        #	SENTENCE_ID\tSENTENCE_TEXT\tNAMED_ENTITY1\tNAMED_ENTITY2
        splittedSentence = line.split("\t")
        if len(splittedSentence) >= 3:
            linesList.append({"content": splittedSentence[1], "namedEntitiesList": splittedSentence[2:]})
        else:
            linesList.append({"content": splittedSentence[1], "namedEntitiesList": []})

    return {"id": doc["id"], "fileId": doc["fileId"], "text": doc["text"], "lines": linesList}
