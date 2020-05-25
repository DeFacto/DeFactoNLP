import os
import jsonlines

from train_sentence_model import get_sentence
from utilities import default_convert

os.chdir('..')

relevant_docs_file = "data/train.jsonl"
new_dataset_file = "sentence_retrieval/coref_dataset.jsonl"
relevant_docs_file = jsonlines.open(relevant_docs_file)
WITH_DOC = False

# with 30000 supported claims and 30000 refuted generated
# 108437    related pairs
# 75798     non-related pairs

# for all the dataset:
# 203932    related pairs
# 144260    non-related pairs

# Augmented Dataset
# 203932    related pairs
# 519793    non-related pairs

claims = []
for line in relevant_docs_file:
    claims.append(line)

with jsonlines.open(new_dataset_file, mode='w') as writer_f:
    neutral = 0
    non_neutral = 0
    for claim in claims:
        if claim['verifiable'] == "NOT VERIFIABLE":
            continue
        line = {'sentence_a': claim['claim']}
        evidences = claim['evidence']
        all_pairs = set()
        for evidence in evidences:
            pairs = set()
            if len(evidence) >= 2:  # needs more than 1 doc to be verifiable
                for e in evidence:
                    pairs.add((str(e[2]), str(e[3])))
            else:
                pairs.add((str(evidence[0][2]), str(evidence[0][3])))
            all_pairs |= pairs
            all_non_related_sentences = set()

        related_sentences = []
        non_related_sentences = []
        for pair in all_pairs:
            related_sentence, non_related_sentence = get_sentence(pair[0], int(pair[1]), more_false=True)
            if WITH_DOC:
                doc = default_convert(pair[0])
                related_sentence = doc + " " + related_sentence
                for i in range(len(non_related_sentence)):
                    non_related_sentence[i] = doc + " " + non_related_sentence[i]
            if related_sentence == "-1" or related_sentence in related_sentences:
                # page failed to load
                continue
            related_sentences.append(related_sentence)
            for sent in non_related_sentence:
                if sent == "ERROR404" or sent in related_sentences \
                        or sent in non_related_sentences:
                    continue
                non_related_sentences.append(sent)

        for related_sentence in related_sentences:
            line['sentence_b'] = related_sentence
            line['stance'] = 1  # related
            writer_f.write(line)
            neutral += 1

        for non_related_sentence in non_related_sentences:
            line['sentence_b'] = non_related_sentence
            line['stance'] = 0  # non-related
            writer_f.write(line)
            non_neutral += 1

print(neutral)
print(non_neutral)
