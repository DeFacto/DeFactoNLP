import jsonlines
import codecs
import json
import numpy as np
from sklearn.externals import joblib
from proof_extraction_train import _extract_features
from defacto.model_nl import ModelNL

wiki_split_docs_dir = "../wiki-pages-split"

document_results_file = "data/dev_relevant_docs.jsonl" # file with tfidf only
# document_results_file = "data/dev_concatenation.jsonl"  # file with tfidf and ner predicted_sentences
document_results_file_oie = "data/dev_concatenation_oie.jsonl"  # file with tfidf and oie

document_results_file = jsonlines.open(document_results_file)
document_results_file_oie = jsonlines.open(document_results_file_oie)

relevant_sent_file = "data/dev_concatenation_oie_sentence.jsonl"

defacto_clf = joblib.load('defacto/defacto_models/rfc.mod')


def get_file(doc):
    try:
        file = codecs.open(wiki_split_docs_dir + "/" + doc + ".json", "r", "latin-1")
        file = json.load(file)
        return file
    except Exception as e:
        print("Failed Loading" + str(doc))
        return ""


def get_lines(file):
    full_lines = file["lines"]
    lines = []
    for _line in full_lines:
        lines.append(_line['content'])
    return lines


def get_pairs_from_doc(doc):
    file = get_file(doc)
    if file == "":
        return ""
    lines = get_lines(file)
    _pairs = []
    for i in range(len(lines)):
        if lines[i] != "":
            _pairs.append((doc, i))
    return _pairs


def get_sentence(doc, line_num):
    file = get_file(doc)
    if file == "":
        return ""
    lines = get_lines(file)
    _sentence = lines[line_num]
    return _sentence


claims = []
for line in document_results_file:
    claims.append(line)

claims_oie = []
for line in document_results_file_oie:
    claims_oie.append(line)

errors = 0
correct = 0
no_prediction = 0
with jsonlines.open(relevant_sent_file, mode='w') as writer_c:
    for line in claims_oie:
        correct_sentences = set()
        flag = False
        try:
            defactoModel = None
            # TODO:get sentence through documents for OIE!
            all_pairs = line['predicted_sentences']
            all_pairs = [tuple(l) for l in all_pairs]
            if 'predicted_pages_oie' in line:
                documents = line['predicted_pages_oie']
                for doc in documents:
                    pairs = get_pairs_from_doc(doc)
                    all_pairs.extend(pairs)
            all_pairs = list(set(all_pairs))
            for pair in all_pairs:
                if defactoModel is None:
                    defactoModel = ModelNL(claim=line['claim'])
                sentence = get_sentence(pair[0], pair[1])
                if sentence == "":
                    continue
                try:
                    x = _extract_features(sentence, line['claim'], defactoModel.triples)
                    x = np.asarray(x)
                    x = x.reshape(1, -1)
                    y = defacto_clf.predict(x)
                    defacto_class = y[0]
                except Exception as e:
                    errors += 1
                    print("Error: " + str(errors))
                if defacto_class == 0:
                    continue
                else:
                    correct_sentences.add(pair)
        except Exception as e:
            print("Error")
            flag = True
        correct += 1
        print(correct)
        if len(correct_sentences) > 0:
            correct_sentences = list(correct_sentences)
            line['predicted_sentences_triple'] = correct_sentences
        else:
            print("NO PREDICTION!!!!")
            if flag:
                no_prediction += 1
            line['predicted_sentences_triple'] = all_pairs
        writer_c.write(line)
print(no_prediction)
print(correct)
print(no_prediction)
