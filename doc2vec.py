import os
import jsonlines
from random import shuffle
import gensim

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# full text and processed in ['text'] tag
wiki_folder = "../wiki-pages-split"
files = os.listdir(wiki_folder)
shuffle(files)

counter = 0
max_counter = 1000000 #1 000 000
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
        print(counter)

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=20, epochs=2)
model.build_vocab(train_text)

model.train(train_text, total_examples=model.corpus_count, epochs=model.epochs)

sentence = "Obama was president of United States of America similar to a Portuguese kind called D. Afonso Henriques"
test_sentence = gensim.utils.simple_preprocess(sentence)
inferred_vector = model.infer_vector(test_sentence)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
print(sims)

file_content = jsonlines.open(wiki_folder + "/" + sims[0][0])
file_content = file_content.read()
text = file_content['text']
print(text)
