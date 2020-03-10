import jsonlines
import json
import doc_retrieval
import sentence_retrieval
import rte.rte as rte
import spacy
import os
import codecs
import unicodedata as ud
import sys

if len(sys.argv)-1 == 3:
    test_file = sys.argv[1]
    results_file = sys.argv[2]
    concatenate_file = sys.argv[3]
else:
    print("#" * 10)
    print("Parameters should be:\n test_file\n results_file \n concatenate_file\nDefaults being used\n")
    print("#" * 10)
    test_file = "data/subsample_train_relevant_docs.jsonl"
    results_file = "predictions_sanity.jsonl"
    concatenate_file = "data/subsample_train_concatenation.jsonl"

nlp = spacy.load('en_core_web_lg')

wiki_dir = "data/wiki-pages/wiki-pages"
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

with jsonlines.open(results_file, mode='w') as writer_r, \
     jsonlines.open(concatenate_file, mode='w') as writer_c:
    for example in test_set:
        relevant_docs, entities = doc_retrieval.getRelevantDocs(example['claim'], wiki_entities, "StanfordNER", nlp)#"spaCy", nlp)#
        relevant_docs = list(set(relevant_docs))
        print(example['claim'])
        print(relevant_docs)
        print(entities)
        relevant_sentences = sentence_retrieval.getRelevantSentences(relevant_docs, entities, wiki_split_docs_dir)
        print(relevant_sentences)
        for i in range(len(example['predicted_sentences'])):

            # load document from TF-IDF
            relevant_doc = ud.normalize('NFC', example['predicted_sentences'][i][0])
            relevant_doc = relevant_doc.replace("/", "-SLH-")
            file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".json", "r", "utf-8")
            file = json.load(file)
            full_lines = file["lines"]

            # load every line of document
            lines = []
            for line in full_lines:
                lines.append(line['content'])
            # file = codecs.open(wiki_split_docs_dir + "/" + relevant_doc + ".txt","r","utf-8")
            # lines = file.readlines()

            # get the specific line
            lines[example['predicted_sentences'][i][1]] = lines[example['predicted_sentences'][i][1]].strip()
            lines[example['predicted_sentences'][i][1]] = lines[example['predicted_sentences'][i][1]].replace("-LRB-",
                                                                                                              " ( ")
            lines[example['predicted_sentences'][i][1]] = lines[example['predicted_sentences'][i][1]].replace("-RRB-",
                                                                                                              " ) ")
            if lines[example['predicted_sentences'][i][1]] == "":
                continue

            # save in a dictionary information of the sentence
            temp = {'id': relevant_doc,
                    'line_num': example['predicted_sentences'][i][1],
                    'sentence': lines[example['predicted_sentences'][i][1]]
                    }
            relevant_sentences.append(temp)
        print(relevant_sentences)
        relevant_docs = relevant_docs + list(example['predicted_pages'])
        relevant_docs = list(set(relevant_docs))
        print("DOCS: ")
        print(relevant_docs)
        result = rte.textual_entailment_evidence_retriever(example['claim'], relevant_sentences, claim_id)
        claim_id = claim_id + 1
        final_result = {'id': example['id'],
                        'predicted_label': result['label']
                        }
        predicted_evidence = []
        for evidence in result['evidence']:
            if [evidence['id'], evidence['line_num']] not in predicted_evidence:
                predicted_evidence.append([evidence['id'], evidence['line_num']])
        final_result['predicted_evidence'] = predicted_evidence
        final_retrieval = example.copy()

        # introduce extraction information performed by NER
        example['predicted_pages_ner'] = relevant_docs
        example['predicted_sentences_final'] = predicted_evidence

        # save info of predictions based on concatenation
        writer_c.write(example)
        writer_r.write(final_result)
