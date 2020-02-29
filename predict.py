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

nlp = spacy.load('en_core_web_lg')

test_file = "data/dev.jsonl"
results_file = "predictions/predictions_dev_sanity.jsonl"

wiki_dir = 'data/wiki-pages/wiki-pages'
wiki_split_docs_dir = "../wiki-pages-split"

test_file = jsonlines.open(test_file)
test_set = []

claim_id = 1

wiki_entities = os.listdir(wiki_split_docs_dir)
for i in range(len(wiki_entities)):
    wiki_entities[i] = wiki_entities[i].replace("-SLH-", "/")
    wiki_entities[i] = wiki_entities[i].replace("_", " ")
    wiki_entities[i] = wiki_entities[i][:-5]
    wiki_entities[i] = wiki_entities[i].replace("-LRB-", "(")
    wiki_entities[i] = wiki_entities[i].replace("-RRB-", ")")

for lines in test_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    test_set.append(lines)

with jsonlines.open(results_file, mode='w') as writer:
    for example in test_set:
        relevant_docs, entities = doc_retrieval.getRelevantDocs(example['claim'], wiki_entities, "spaCy", nlp)
        relevant_docs = list(set(relevant_docs))
        print(example['claim'])
        print(relevant_docs)
        print(entities)
        relevant_sentences = sentence_retrieval.getRelevantSentences(relevant_docs, entities, wiki_split_docs_dir)
        print(relevant_sentences)
        print(example)
        example['predicted_sentences'] = relevant_sentences
        example['predicted_pages'] = relevant_docs
        for i in range(len(example['predicted_sentences'])):
            print(example['predicted_sentences'][i]['id'])
            relevant_doc = ud.normalize('NFC', example['predicted_sentences'][i]['id'])
            relevant_doc = relevant_doc.replace("/", "-SLH-")
            file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".json", "r", "utf-8")
            file = json.load(file)
            full_lines = file["lines"]
            lines = []
            for line in full_lines:
                lines.append(line['content'])
            # file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
            # lines = file.readlines()
            print(full_lines)
            print(lines)
            lines[example['predicted_sentences'][i]['line_num']] = lines[example['predicted_sentences'][i]['line_num']].strip()
            lines[example['predicted_sentences'][i]['line_num']] = lines[example['predicted_sentences'][i]['line_num']].replace("-LRB-",
                                                                                                              " ( ")
            lines[example['predicted_sentences'][i]['line_num']] = lines[example['predicted_sentences'][i]['line_num']].replace("-RRB-",
                                                                                                              " ) ")
            if lines[example['predicted_sentences'][i]['line_num']] == "":
                continue
            temp = {'id': relevant_doc, 'line_num': example['predicted_sentences'][i]['line_num'],
                    'sentence': lines[example['predicted_sentences'][i]['line_num']]}
            relevant_sentences.append(temp)

        print(relevant_sentences)
        relevant_docs = relevant_docs + list(example['predicted_pages'])
        relevant_docs = list(set(relevant_docs))
        if len(relevant_sentences) == 0:
            writer.write(example)
            continue
        result = rte.textual_entailment_evidence_retriever(example['claim'], relevant_sentences, claim_id)
        print("\n##########")
        print(result)
        print("##########\n")
        claim_id += 1
        final_result = {'id': example['id'], 'predicted_label': result['label']}
        predicted_evidence = []
        for evidence in result['evidence']:
            if [evidence['id'], evidence['line_num']] not in predicted_evidence:
                predicted_evidence.append([evidence['id'], evidence['line_num']])
        final_result['predicted_evidence'] = predicted_evidence
        writer.write(final_result)
