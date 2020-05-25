import os
import jsonlines
from random import shuffle
import gensim
import sys
from scipy import spatial
import spacy
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords

fname = "word2vec.model"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

spacy_nlp = spacy.load('en_core_web_sm')

if len(sys.argv) - 1 == 1:
    max_counter = int(sys.argv[1])
    if max_counter == -1:
        max_counter = 9999999
else:
    max_counter = 10000  # 10 000
    print("Max Counter not defined!")
    print("Set Default Value: " + str(max_counter))

# full text and processed in ['text'] tag
wiki_folder = "../wiki-pages-split"
files = os.listdir(wiki_folder)
shuffle(files)

counter = 0

train_sentences = []
tokens = []
for file in files:
    _name = file[:-5]
    title = _name.replace("_", " ")
    tokens = gensim.utils.simple_preprocess(title)
    train_sentences.append(tokens)

    file_content = jsonlines.open(wiki_folder + "/" + file)
    lines = file_content.read()['lines']
    for line in lines:
        if len(line['content']) < 2:
            continue
        tokens = gensim.utils.simple_preprocess(line['content'])
        train_sentences.append(tokens)
    if counter > max_counter:
        # adding required docs by fever with the claim given
        file_content = jsonlines.open(wiki_folder + "/" + "Telemundo.json")
        lines = file_content.read()['lines']
        for line in lines:
            print(line['content'])
            tokens = gensim.utils.simple_preprocess(line['content'])
            train_sentences.append(tokens)

        file_content = jsonlines.open(wiki_folder + "/" + "Hispanic_and_Latino_Americans.json")
        lines = file_content.read()['lines']
        for line in lines:
            print(line['content'])
            tokens = gensim.utils.simple_preprocess(line['content'])
            train_sentences.append(tokens)

        break
    else:
        counter += 1
        if counter % 1000 == 0:
            print(counter)

model = gensim.models.Word2Vec(iter=1, min_count=5, size=500, workers=4)  # an empty model, no training yet
model.build_vocab(train_sentences)  # can be a non-repeatable, 1-pass generator

print(model.epochs)
model.train(train_sentences, total_examples=model.corpus_count, epochs=30)  # can be a non-repeatable, 1-pass generator
index2word_set = set(model.wv.index2word)


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def word2vec(text1, text2):
    # print(text1)
    # print(text2)
    s1_afv = avg_feature_vector(text1, model=model, num_features=500, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(text2, model=model, num_features=500, index2word_set=index2word_set)
    if np.sum(s1_afv) == 0 or np.sum(s2_afv) == 0:
        return 0
        # text1 = spacy_nlp(text1)
        # text2 = spacy_nlp(text2)
        # sim = text1.similarity(text2)
    else:
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)

    return sim


sentence = "Telemundo is a English-language television network."
document = "Telemundo"
document_2 = "Hispanic_and_Latino_Americans"

print(word2vec("cat", "man"))
print(word2vec("cat", "dog"))
#print(model.similarity("cat", "man"))
#print(model.similarity("cat", "dog"))

tokens_sentence = gensim.utils.simple_preprocess(sentence)
sentence = ' '.join(map(str, tokens_sentence))
print(sentence)

best = [0, 0, 0, 0, 0]
docs = ["", "", "", "", ""]
for file in files:
    _name = file[:-5]
    title = _name.replace("_", " ")
    tokens = gensim.utils.simple_preprocess(title)
    text = ' '.join(map(str, tokens))
    sim = word2vec(sentence, text)
    if sim > best[0]:
        best[0] = sim
        docs[0] = _name

    elif sim > best[1]:
        best[1] = sim
        docs[1] = _name

    elif sim > best[2]:
        best[2] = sim
        docs[2] = _name

    elif sim > best[3]:
        best[3] = sim
        docs[3] = _name

    elif sim > best[4]:
        best[4] = sim
        docs[4] = _name

print(best)
print(docs)

tokens_sentence = gensim.utils.simple_preprocess(sentence)

for token_sentence in tokens_sentence:
    best = [0, 0, 0, 0, 0]
    docs = ["", "", "", "", ""]
    for file in files:
        _name = file[:-5]
        title = _name.replace("_", " ")
        tokens = gensim.utils.simple_preprocess(title)
        text = ' '.join(map(str, tokens))
        sim = word2vec(token_sentence, text)
        if sim > best[0]:
            best[0] = sim
            docs[0] = _name

        elif sim > best[1]:
            best[1] = sim
            docs[1] = _name

        elif sim > best[2]:
            best[2] = sim
            docs[2] = _name

        elif sim > best[3]:
            best[3] = sim
            docs[3] = _name

        elif sim > best[4]:
            best[4] = sim
            docs[4] = _name
    print(best)
    print(docs)

print(word2vec("telemundo", sentence))

model.save('models/word2vec')
