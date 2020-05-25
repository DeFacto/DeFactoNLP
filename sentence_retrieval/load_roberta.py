import jsonlines
import torch
import os

os.chdir('..')

from run_sentence_selection import get_sentence, clean_sentence
from triple_sentence_selection import get_pairs_from_doc

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()

if __name__ == "__main__":
    # a = 'Roberta is a heavily optimized version of BERT.'
    # b = 'Roberta is not very optimized.'
    # c = 'Roberta is based on BERT.'
    # d = 'Roberta took a walk into the woods.'
    #
    # tokens = roberta.encode(a, b)
    # print(roberta.predict('mnli', tokens))

    wiki_split_docs_dir = "../wiki-pages-split"
    relevant_docs_file = "data/dev_oie_test.jsonl"
    relevant_sent_file = "data/dev_test_sentence_baby.jsonl"

    relevant_docs_file = jsonlines.open(relevant_docs_file)

    claims = []
    for line in relevant_docs_file:
        claims.append(line)

    STOP = -1
    with jsonlines.open(relevant_sent_file, mode='w') as writer_c:
        for claim in claims:
            print(claim['claim'])
            print(STOP)
            # get all possible sentences
            pair_sent_pair = {}

            # Adding TF-IDF Sentences!
            all_pairs = claim['predicted_sentences']
            all_pairs = [tuple(pair) for pair in all_pairs]
            for pair in all_pairs:
                sentence = get_sentence(pair[0], pair[1])
                sentence = clean_sentence(sentence)
                pair_sent_pair[sentence] = (pair[0], pair[1])
            print(len(claim['predicted_pages_oie']))
            print(len(all_pairs))
            for doc in claim['predicted_pages_oie']:
                pairs = get_pairs_from_doc(doc)
                for pair in pairs:
                    sentence = get_sentence(pair[0], pair[1])
                    sentence = clean_sentence(sentence)
                    pair_sent_pair[sentence] = (pair[0], pair[1])

            for pair in claim['predicted_sentences']:
                sentence = get_sentence(pair[0], pair[1])
                sentence = clean_sentence(sentence)
                pair_sent_pair[sentence] = (pair[0], pair[1])

            sentences_predicted = []
            for sentence in pair_sent_pair:
                try:
                    tokens = roberta.encode(claim['claim'], sentence)
                    prediction = roberta.predict('mnli', tokens).argmax().item()
                    if prediction == 0 or prediction == 2:
                        sentences_predicted.append(pair_sent_pair[sentence])
                except Exception as e:
                    print(e)

            claim['predicted_sentences_roberta'] = sentences_predicted
            writer_c.write(claim)

            STOP -= 1
            if STOP == 0:
                break
