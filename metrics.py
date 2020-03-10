import jsonlines
import sys
from scorer import fever_score

train_file = "data/subsample_train.jsonl"
train_relevant_file = "data/subsample_train_relevant_docs.jsonl"
train_concatenate_file = "data/subsample_train_concatenation.jsonl"
train_predictions_file = "predictions/predictions_train.jsonl"

train_file = jsonlines.open(train_file)
train_relevant_file = jsonlines.open(train_relevant_file)
train_concatenate_file = jsonlines.open(train_concatenate_file)
train_predictions_file = jsonlines.open(train_predictions_file)

train_set = []
train_relevant = []
train_concatenate = []
train_prediction = []

for lines in train_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    train_set.append(lines)

for lines in train_relevant_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    train_relevant.append(lines)

for lines in train_concatenate_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    train_concatenate.append(lines)

# this evidence addition is irrelevant
info_by_id = dict((d['id'], dict(d, index=index)) for (index, d) in enumerate(train_set))
for lines in train_predictions_file:
    lines['evidence'] = info_by_id.get(lines['id'])['evidence']
    train_prediction.append(lines)

# All claims
stop = 0

# List with dicts with all important data
'''
id : id of the claim
verifiable : boolean of 1 and 0 with respective meaning
docs : set of documents that verify the claim
docs_sep : set of documents separated
evidences: list of tuples of <doc, line>
difficulties: list of the number of sentences needed to be evidence
'''
gold_data = []

for claim in train_set:

    # init gold dict
    gold_dict = {'id': claim['id']}

    if claim['verifiable'] == "VERIFIABLE":
        gold_dict['verifiable'] = 1
    else:
        gold_dict['verifiable'] = 0

    # get gold inputs
    gold_documents = set()
    gold_documents_separated = set()
    sentences_pair = set()
    evidences = claim['evidence']
    difficulties = []
    for evidence in evidences:
        doc_name = ''
        difficulty = 0
        if len(evidence) > 1:  # needs more than 1 doc to be verifiable
            for e in evidence:
                doc_name += str(e[2])
                doc_name += " "
                sentences_pair.add((str(e[2]), str(e[3])))  # add gold sentences
                gold_documents_separated.add(str(e[2]))  # add the document
                difficulty += 1
            doc_name = doc_name[:-1]  # erase the last blank space
        else:
            doc_name = str(evidence[0][2])
            gold_documents_separated.add(str(evidence[0][2]))
            sentences_pair.add((str(evidence[0][2]), str(evidence[0][3])))
            difficulty = 1
        difficulties.append(difficulty)
        gold_documents.add(doc_name)
    gold_dict['difficulties'] = difficulties
    gold_dict['docs'] = gold_documents
    gold_dict['evidences'] = sentences_pair
    gold_dict['docs_sep'] = gold_documents_separated

    gold_data.append(gold_dict)

    # flag to stop if needed
    stop += 1
    if stop == -1:
        break

gold_data = dict((item['id'], item) for item in gold_data)

stop = 0

doc_found = 0
doc_noise = 0
gold_doc_found = 0
gold_doc_not_found = 0

precision_correct = 0
precision_incorrect = 0
recall_correct = 0
recall_incorrect = 0
specificity = 0

precision_sent_correct = 0
precision_sent_incorrect = 0
recall_sent_correct = 0
recall_sent_incorrect = 0
sent_found = 0
sent_found_if_doc_found = 0

total_claim = 0
for claim in train_relevant:
    _id = claim['id']
    gold_dict = gold_data.get(_id)

    # no search is needed... no information on gold dict about retrieval
    if not gold_dict['verifiable']:
        continue

    # document analysis
    # TODO: Analyse NER and TF-IDF
    doc_correct = 0
    doc_incorrect = 0
    gold_incorrect = 0
    docs = set()
    gold_docs = gold_dict['docs_sep']

    for doc in claim['predicted_pages']:
        if doc in gold_docs:
            doc_correct += 1
        else:
            doc_incorrect += 1
        docs.add(doc)

    precision_correct += doc_correct / len(docs)
    precision_incorrect += doc_incorrect / len(docs)
    recall_correct += doc_correct / len(gold_docs)
    recall_incorrect += doc_incorrect / len(gold_docs)

    for gold_doc in gold_docs:
        if gold_doc not in docs:
            gold_incorrect += 1

    specificity += gold_incorrect / len(gold_docs)

    if doc_correct > 0:
        doc_found += 1

    # sentence analysis TODO: check sentences
    sentences = set()
    for sent in claim['predicted_sentences']:
        sentences.add((str(sent[0]), str(sent[1])))

    evidences = gold_dict['evidences']
    sent_correct = 0
    sent_incorrect = 0
    flag = False
    for sent in sentences:
        if sent in evidences:
            sent_correct += 1
            flag = True
        else:
            sent_incorrect += 1

    if flag:
        sent_found += 1

    if doc_correct and flag:
        sent_found_if_doc_found += 1

    precision_sent_correct += sent_correct / len(sentences)
    precision_sent_incorrect += sent_incorrect / len(sentences)
    recall_sent_correct += sent_correct / len(evidences)
    recall_sent_incorrect += sent_incorrect / len(evidences)

    # TODO: create all possible pair in order to see if it appears in gold_dict['docs']
    # claim['predicted_sentences']

    # flag to stop if needed
    total_claim += 1
    stop += 1
    if stop == -1:
        break

precision_correct /= total_claim
precision_incorrect /= total_claim
recall_correct /= total_claim
recall_incorrect /= total_claim
specificity /= total_claim
doc_found /= total_claim

print("\n#############")
print("# DOCUMENTS #")
print("#############")
print("Precision (Document Retrieved):\t\t\t\t\t\t " + str(precision_correct))  # precision
print("Fall-out (incorrect documents):\t\t\t\t\t\t " + str(precision_incorrect))  # precision
print("Recall (Relevant Documents):\t\t\t\t\t\t " + str(recall_correct))  # recall
print("Percentage of gold documents NOT found:\t\t\t\t " + str(recall_incorrect))  # recall
print("Fall-out: " + str(specificity))
print("Percentage of at least one document found correctly: " + str(doc_found))  # recall

precision_sent_correct /= total_claim
precision_sent_incorrect /= total_claim
recall_sent_correct /= total_claim
recall_sent_incorrect /= total_claim
sent_found /= total_claim
sent_found_if_doc_found /= total_claim
another_sent = sent_found_if_doc_found / doc_found

print("\n#############")
print("# SENTENCES #")
print("#############")
print("Precision (Sentences Retrieved):\t\t\t\t\t " + str(precision_sent_correct))  # precision
print("Precision (incorrect Sentences):\t\t\t\t\t " + str(precision_sent_incorrect))  # precision
print("Recall (Relevant Sentences):\t\t\t\t\t\t " + str(recall_sent_correct))  # recall
print("Percentage of gold Sentences NOT found:\t\t\t\t " + str(recall_sent_incorrect))  # recall
print("Percentage of at least one Sentence found correctly: " + str(sent_found))  # recall
print("Percentage of at least one Sentence found correctly: " + str(sent_found_if_doc_found))  # recall
print("Percentage of at least one Sentence found correctly: " + str(another_sent))  # recall

# scores from fever
results = fever_score(train_prediction, actual=train_set)

print("\n#########")
print("# FEVER #")
print("#########")
print("Strict_score: \t\t" + str(results[0]))
print("Acc_score: \t\t\t" + str(results[1]))
print("Precision: \t\t\t" + str(results[2]))
print("Recall: \t\t\t" + str(results[3]))
print("F1-Score: \t\t\t" + str(results[4]))
