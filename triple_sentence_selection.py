import jsonlines
import codecs
import json
import numpy as np
from sklearn.externals import joblib
from proof_extraction_train import _extract_features
from defacto.model_nl import ModelNL
import unicodedata as ud

# Non-CoRef Docs
# wiki_split_docs_dir = "../wiki-pages-split"
# CoRef Docs
wiki_split_docs_dir = "../wiki-pages-coref"


def clean_doc(_doc):
    _doc = _doc.replace("-SLH-", " / ")
    _doc = _doc.replace("_", " ")
    _doc = _doc.replace("-LRB-", " ( ")
    _doc = _doc.replace("-RRB-", " ) ")
    _doc = _doc.replace("-COLON-", " : ")
    return _doc


def get_file(_doc, _wiki_split_docs_dir=wiki_split_docs_dir):
    try:
        _doc = ud.normalize('NFC', _doc)
        file = codecs.open(_wiki_split_docs_dir + "/" + _doc + ".json", "r", "utf8")
        file = json.load(file)
        return file
    except Exception as _e:
        print("Failed Loading" + str(_doc) + str(_e))
        return ""


def get_lines(file):
    full_lines = file["lines"]
    lines = []
    for _line in full_lines:
        lines.append(_line['content'])
    return lines


def get_pairs_from_doc(_doc, _wiki_split_docs_dir=wiki_split_docs_dir):
    file = get_file(_doc, _wiki_split_docs_dir=_wiki_split_docs_dir)
    if file == "":
        return ""
    lines = get_lines(file)
    _pairs = []
    for i in range(len(lines)):
        if lines[i] != "":
            _pairs.append((_doc, i))
    return _pairs


def get_sentence(_doc, line_num, _wiki_split_docs_dir=wiki_split_docs_dir):
    file = get_file(_doc, _wiki_split_docs_dir=_wiki_split_docs_dir)
    if file == "":
        return ""
    lines = get_lines(file)
    _sentence = lines[line_num]
    return _sentence


if __name__ == "__main__":
    document_results_file = "data/dev_test_triple_baby.jsonl"  # file with tfidf only
    # document_results_file = "data/dev_concatenation.jsonl"  # file with tfidf and ner predicted_sentences
    document_results_file_oie = "data/dev_test_triple_baby.jsonl"  # file with tfidf and oie

    document_results_file = jsonlines.open(document_results_file)
    document_results_file_oie = jsonlines.open(document_results_file_oie)

    results_file = "data/dev_coref_triple.jsonl"

    defacto_clf = joblib.load('defacto/defacto_models/rfc.mod')

    claims = []
    for line in document_results_file:
        claims.append(line)

    claims_oie = []
    for line in document_results_file_oie:
        claims_oie.append(line)

    errors = 0
    correct = 0
    no_prediction = 0

    with jsonlines.open(results_file, mode='w') as writer_c:
        advance_lines = 0
        for line in claims_oie:
            advance_lines -= 1
            if advance_lines > 0:
                continue
            correct_sentences = set()
            correct_probabilities = set()
            flag = False
            try:
                defactoModel = None
                all_pairs = line['predicted_sentences']
                all_pairs = [tuple(p) for p in all_pairs]
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
                        y_prob = defacto_clf.predict_proba(x)
                        defacto_class = y[0]
                    except Exception as e:
                        errors += 1
                        print("Error: " + str(errors))
                    if defacto_class == 0:
                        continue
                    else:
                        correct_sentences.add(pair)
                        correct_probabilities.add((pair[0], pair[1], y_prob[0][1]))
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
                line['predicted_sentences_triple'] = line['predicted_sentences']

            # Create exactly 10 sentences for the triple selection
            size = len(line['predicted_sentences_triple'])
            ordered_sents = sorted(correct_probabilities, key=lambda tup: tup[2], reverse=True)
            BREAK_INFINITE_LOOP = 10
            line['predicted_sentences_triple_all'] = line['predicted_sentences_triple']
            while size != 10:
                if size > 10:
                    line['predicted_sentences_triple'] = []
                    for i in range(0, 10):
                        line['predicted_sentences_triple'].append((ordered_sents[i][0], ordered_sents[i][1]))
                        print(ordered_sents[i][2])
                elif size < 10:
                    for pair in all_pairs:
                        if pair not in line['predicted_sentences_triple'] and pair not in correct_sentences:
                            line['predicted_sentences_triple'].append(pair)
                            break
                BREAK_INFINITE_LOOP -= 1
                if BREAK_INFINITE_LOOP < 1:
                    print("INFINITE LOOP DETECTED")
                    break
                size = len(line['predicted_sentences_triple'])
            # for pair in line['predicted_sentences_triple']:
            #     print(pair)
            writer_c.write(line)
    print(no_prediction)
    print(correct)
    print(no_prediction)
