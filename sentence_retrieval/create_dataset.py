import os
import jsonlines

from train_sentence_model import get_sentence

os.chdir('..')

relevant_docs_file = "data/train_verifiable.jsonl"
new_dataset_file = "sentence_retrieval/train_sentence_retrieval.jsonl"
relevant_docs_file = jsonlines.open(relevant_docs_file)

claims = []
for line in relevant_docs_file:
    claims.append(line)

with jsonlines.open(new_dataset_file, mode='w') as writer_f:
    neutral = 0
    non_neutral = 0
    for claim in claims:
        line = {'sentence_a': claim['claim']}
        evidences = claim['evidence']
        for evidence in evidences:
            line = {}
            pairs = set()
            if len(evidence) > 1:  # needs more than 1 doc to be verifiable
                for e in evidence:
                    pairs.add((str(e[2]), str(e[3])))
            else:
                pairs.add((str(evidence[0][2]), str(evidence[0][3])))
            all_non_related_sentences = set()
            for pair in pairs:
                sentence_b, non_related_sentences = get_sentence(pair[0], int(pair[1]))
                if sentence_b == "-1":
                    # page failed to load
                    continue
                line['sentence_b'] = sentence_b
                line['stance'] = 1  # related
                writer_f.write(line)
                neutral += 1
                if non_related_sentences != "ERROR404":
                    line['sentence_b'] = non_related_sentences
                    line['stance'] = 0  # non-related
                    writer_f.write(line)
                    non_neutral += 1

print(neutral)
print(non_neutral)
