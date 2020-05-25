import codecs
import json
from urllib import parse

import spacy
import neuralcoref
import os

from run_sentence_selection import clean_sentence

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

wiki_split_docs_dir = "../wiki-pages-split"
wiki_coref_docs_dir = "../wiki-pages-coref"
wiki_files = os.listdir(wiki_split_docs_dir)
# wiki_files = ["Portugal.json"]
counter = 0

# Examples number...320000
# Battle_of_Tanlwe_Chaung.json
wiki_files = wiki_files[3610000:]
for wiki_file in wiki_files:
    try:
        file = codecs.open(wiki_split_docs_dir + "/" + wiki_file, "r", "utf-8")
    except Exception:
        print("Failed Loading: " + str(wiki_file))
        continue
    doc = json.load(file)
    full_lines = doc["lines"]
    lines = []
    full_text = ""
    for line in full_lines:
        sentence = clean_sentence(line['content'])
        lines.append(sentence)
        full_text += sentence
        full_text += "!. "
    full_text = full_text[:-3]

    coref_sentence = nlp(full_text)
    # print(full_text)
    new_sentence = coref_sentence._.coref_resolved
    new_sentences = new_sentence.split("!.")

    while len(full_lines) > len(new_sentences):
        new_sentences.append(" ")

    for i in range(len(full_lines)):
        # print(full_lines[i]['content'])
        # print(new_sentences[i][1:])
        # print("-----------")
        try:
            if new_sentences[i][0] == " ":
                full_lines[i]['content'] = new_sentences[i][1:]
            else:
                full_lines[i]['content'] = new_sentences[i]
        except Exception as e:
            print(e)
            print(wiki_file)
            print(len(full_lines))
            print(len(new_sentences))
            continue
    doc["lines"] = full_lines
    new_file = codecs.open(wiki_coref_docs_dir + "/" + wiki_file, "w+", "utf-8")
    json.dump(doc, new_file)
    new_file.close()

    if counter % 10000 == 0:
        print("Examples number...{}".format(counter))
        print(wiki_file)
    counter += 1

print(wiki_files[0])
url = ""
url = parse.unquote(url)
# File input:
# "test"
# "lines"
#  # "content"
#  # "namedEntitiesList"
# doc = nlp('She has a dog. Paula loves him.')
doc = nlp('Paulo has a dog .!.!. He loves him .!.!. The dog studies for him')

print(doc._.has_coref)
print(doc._.coref_clusters)
print(doc)
print(doc._.coref_resolved)
