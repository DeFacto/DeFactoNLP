import os
import jsonlines
from random import shuffle
import gensim
import sys
import spacy
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import Doc2Vec

fname = "doc2vec.model"

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

spacy_nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

customize_stop_words = [
    "-LRB-", "-RRB-", "-LSB-", "-LRB-"
]

for w in customize_stop_words:
    spacy_nlp.vocab[w].is_stop = True

if len(sys.argv) - 1 == 1:
    max_counter = int(sys.argv[1])
else:
    max_counter = 10000  # 10 000
    print("Max Counter not defined!")
    print("Set Default Value: " + str(max_counter))


def pre_process(doc):
    # doc = spacy_nlp(doc)

    # lemma_tokens = [token.lemma_ for token in doc] 
    # doc = ' '.join(map(str, lemma_tokens))
    # doc = spacy_nlp(doc)

    # tokens = [token.text for token in doc if not token.is_stop]

    # text = ' '.join(map(str, tokens))
    text = remove_stopwords(doc)
    return text

# TODO:Remove all STOP-WORDS and Lemmatize every token!!!!!

# full text and processed in ['text'] tag
wiki_folder = "data/wiki-pages-split"
files = os.listdir(wiki_folder)
shuffle(files)

counter = 0

train_text = []
tokens = []
for file in files:
    file_content = jsonlines.open(wiki_folder + "/" + file)
    doc = file_content.read()['text']
    text = pre_process(doc)

    if counter > max_counter:
        # adding required docs by fever with the claim given
        file_content = jsonlines.open(wiki_folder + "/" + "Telemundo.json")
        doc = file_content.read()['text']
        text = pre_process(doc)
        tokens = gensim.utils.simple_preprocess(text)
        print(tokens)
        train_text.append(gensim.models.doc2vec.TaggedDocument(tokens, ["Telemundo.json"]))

        file_content = jsonlines.open(wiki_folder + "/" + "Hispanic_and_Latino_Americans.json")
        doc = file_content.read()['text']
        text = pre_process(doc)
        tokens = gensim.utils.simple_preprocess(text)
        train_text.append(gensim.models.doc2vec.TaggedDocument(tokens, ["Hispanic_and_Latino_Americans.json"]))

        break
    else:
        tokens = gensim.utils.simple_preprocess(text)
        train_text.append(gensim.models.doc2vec.TaggedDocument(tokens, [file]))
        counter += 1
        if counter % 1000 == 0:
            print(counter)

model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
#model = Doc2Vec.load(fname)
model.build_vocab(train_text)#,keep_raw_vocab=True)#, update=True)

model.train(train_text, total_examples=model.corpus_count, epochs=model.epochs)

sentence = "Telemundo is a English-language television network."
text = pre_process(sentence)
tokens = gensim.utils.simple_preprocess(text)
print(tokens)
for token in tokens:
    print(token)
    inferred_vector = model.infer_vector([token])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    STOP = 3
    for doc, sim in sims:
        file_content = jsonlines.open(wiki_folder + "/" + doc)
        file_content = file_content.read()
        text = file_content['text']
        print("\n" + doc + " -- " + str(sim) + ": \n")  # + text)
        if STOP == 0:
            break
        else:
            STOP -= 1

    for doc, sim in sims:
        if doc != "Hispanic_and_Latino_Americans.json" and doc != "Telemundo.json":
            continue
        print(doc + " -- " + str(sim))
    print("\n")

model.save(fname)

inferred_vector = model.infer_vector(tokens)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

STOP = 3
for doc, sim in sims:
    file_content = jsonlines.open(wiki_folder + "/" + doc)
    file_content = file_content.read()
    text = file_content['text']
    print("\n" + doc + " -- " + str(sim) + ": \n")  # + text)
    if STOP == 0:
        break
    else:
        STOP -= 1

for doc, sim in sims:
    if doc != "Hispanic_and_Latino_Americans.json" and doc != "Telemundo.json":
        continue
    print(doc + " -- " + str(sim))
