import os
import jsonlines
from random import shuffle
import gensim
import sys

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if len(sys.argv) - 1 == 1:
    max_counter = 1000000  # 1 000 000
else:
    max_counter = 10000  # 10 000
    print("Max Counter not defined!")
    print("Set Default Value: " + str(max_counter))

# full text and processed in ['text'] tag
wiki_folder = "../wiki-pages-split"
files = os.listdir(wiki_folder)
shuffle(files)

counter = 0

train_text = []
tokens = []
for file in files:
    file_content = jsonlines.open(wiki_folder + "/" + file)
    file_content = file_content.read()
    text = file_content['text']
    if counter > max_counter:
        break
    else:
        tokens = gensim.utils.simple_preprocess(text)
        train_text.append(gensim.models.doc2vec.TaggedDocument(tokens, [file]))
        counter += 1
        if counter % 1000 == 0:
            print(counter)

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=20, epochs=2)
model.build_vocab(train_text)

model.train(train_text, total_examples=model.corpus_count, epochs=model.epochs)

sentence = "Obama was president of United States of America similar to a Portuguese person called D. Afonso Henriques"
test_sentence = gensim.utils.simple_preprocess(sentence)
inferred_vector = model.infer_vector(test_sentence)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

STOP = 5
for doc, sim in sims:
    file_content = jsonlines.open(wiki_folder + "/" + doc)
    file_content = file_content.read()
    text = file_content['text']
    print("\n" + doc + " -- " + str(sim) + ": \n" + text)
    if STOP == 0:
        break
    else:
        STOP -= 1
