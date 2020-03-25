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
    clauses, ner_spacy = clausie.clausie(claim)
    if len(triples) == 0:
        for clause in clauses:
            for sub in clause['S']:
                ents.add(sub.text)
            for obj in clause['O']:
                ents.add(obj.text)
    print(ner_spacy)
    print(ents)
    for ent in ner_spacy:
        ents.add(ent.text)

    if len(ents) > 5:
        ents = clean_entities(ents)

    ents = list(ents)

    if len(ents) != 0:
        _str = ""
        for ent in ents:
            _str += ent
            _str += " "
        _str = _str[:-1]
        ents.append(_str)

        if "film" in claim:
            _str += " ( film )"
            ents.append(_str)
        elif "(" in claim:
            disambiguation = claim[claim.find("(") + 1:claim.find(")")]
            _str += " " + disambiguation
            ents.append(_str)
    print(ents)
    docs, entities = getClosestDocs(wiki_entities, ents)

    return docs, entities


# getting the 3 closest docs!
def getClosestDocs(wiki_entities, entities):
    entities = list(entities)
    for i in range(len(entities)):
        entities[i] = str(entities[i])
    selected_docs = []
    for ent in entities:
        ent = ud.normalize('NFC', ent)

        best_1 = 1.1
        best_match_1 = ""

        best_2 = 1.1
        best_match_2 = ""

        best_3 = 1.1
        best_match_3 = ""

        for we in wiki_entities:
            dist = stringdist.levenshtein_norm(we, ent)
            if dist < best_1:
                best_1 = dist
                best_match_1 = we

            elif dist < best_2:
                best_2 = dist
                best_match_2 = we

            elif dist < best_3:
                best_3 = dist
                best_match_3 = we

        best_match_1 = best_match_1.replace(" ", "_")
        best_match_1 = best_match_1.replace("/", "-SLH-")
        best_match_1 = best_match_1.replace("(", "-LRB-")
        best_match_1 = best_match_1.replace(")", "-RRB-")

        best_match_2 = best_match_2.replace(" ", "_")
        best_match_2 = best_match_2.replace("/", "-SLH-")
        best_match_2 = best_match_2.replace("(", "-LRB-")
        best_match_2 = best_match_2.replace(")", "-RRB-")

        best_match_3 = best_match_3.replace(" ", "_")
        best_match_3 = best_match_3.replace("/", "-SLH-")
        best_match_3 = best_match_3.replace("(", "-LRB-")
        best_match_3 = best_match_3.replace(")", "-RRB-")

        selected_docs.append(best_match_1)
        selected_docs.append(best_match_2)
        selected_docs.append(best_match_3)
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
