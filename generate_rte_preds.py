import jsonlines
import json
import doc_retrieval
import sentence_retrieval
import rte.rte as rte
import utilities
import spacy
import os
import codecs
import unicodedata as ud

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

relevant_sentences_file = "data/shared_task_dev_public_relevant_docs.jsonl"
instances = []
zero_results = 0

relevant_sentences_file = jsonlines.open(relevant_sentences_file)
model = "rte/fever_output/model.tar.gz"
model = load_archive(model)
predictor = Predictor.from_archive(model)

wiki_dir = "data/wiki-pages/wiki-pages"
wiki_split_docs_dir = "../wiki-pages-split"

claim_num = 1

for line in relevant_sentences_file:
    instances.append(line)


def create_test_set(claim, candidateEvidences, claim_num):
    testset = []
    for elem in candidateEvidences:
        testset.append({"hypothesis": claim, "premise": elem})
    return testset


def run_rte(claim, evidence, claim_num):
    fname = "claim_" + str(claim_num) + ".json"
    test_set = create_test_set(claim, evidence, claim_num)
    preds = predictor.predict_batch_json(test_set)
    return preds


for i in range(len(instances)):
    claim = instances[i]['claim']
    print(claim)
    evidence = instances[i]['predicted_sentences']
    potential_evidence_sentences = []
    for sentence in evidence:
        # print(sentence)
        # print(sentence[0])
        # load document from TF-IDF
        relevant_doc = ud.normalize('NFC', sentence[0])
        relevant_doc = relevant_doc.replace("/", "-SLH-")
        file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".json", "r", "utf-8")
        file = json.load(file)
        full_lines = file["lines"]

        lines = []
        for line in full_lines:
            lines.append(line['content'])

        lines[sentence[1]] = lines[sentence[1]].strip()
        lines[sentence[1]] = lines[sentence[1]].replace("-LRB-", " ( ")
        lines[sentence[1]] = lines[sentence[1]].replace("-RRB-", " ) ")

        potential_evidence_sentences.append(lines[sentence[1]])

    # Just adding a check
    if len(potential_evidence_sentences) == 0:
        zero_results += 1
        potential_evidence_sentences.append("Nothing")
        evidence.append(["Nothing", 0])

    preds = run_rte(claim, potential_evidence_sentences, claim_num)

    saveFile = codecs.open("rte/entailment_predictions/claim_" + str(claim_num) + ".json", mode="w+", encoding="utf-8")
    for i in range(len(preds)):
        # print(preds)
        # print(evidence)
        preds[i]['claim'] = claim
        preds[i]['premise_source_doc_id'] = evidence[i][0]
        preds[i]['premise_source_doc_line_num'] = evidence[i][1]
        preds[i]['premise_source_doc_sentence'] = potential_evidence_sentences[i]
        saveFile.write(json.dumps(preds[i], ensure_ascii=False) + "\n")

    saveFile.close()
    claim_num += 1
    print(claim_num)

print("Number of Zero Sentences Found: " + str(zero_results))
