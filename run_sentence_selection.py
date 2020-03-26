import jsonlines
import codecs
import json
from sentence_transformers import SentenceTransformer
import scipy.spatial
from sklearn.metrics.pairwise import cosine_similarity

wiki_split_docs_dir = "data/wiki-pages-split"
relevant_docs_file = "data/dev_concatenation.jsonl"
relevant_sent_file = "data/dev_sentence_selection.jsonl"

relevant_docs_file = jsonlines.open(relevant_docs_file)


# relevant_sent_file = jsonlines.open(relevant_sent_file)

def get_sentence(doc, line_num):
    try:
        file = codecs.open(wiki_split_docs_dir + "/" + doc + ".json", "r", "latin-1")
    except:
        print("Failed Loading" + str(doc))
        return ""

    file = json.load(file)
    full_lines = file["lines"]
    lines = []
    for line in full_lines:
        lines.append(line['content'])
    sentence = lines[line_num]
    return sentence


def clean_sentence(_sentence):
    _sentence = _sentence.replace("-LRB-", "(")
    _sentence = _sentence.replace("-RRB-", ")")
    _sentence = _sentence.replace("-LSB-", "[")
    _sentence = _sentence.replace("-RSB-", "]")
    return _sentence


# model = SentenceTransformer('bert-base-nli-mean-tokens')
embedder = SentenceTransformer('bert-large-nli-mean-tokens')
# embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

claims = []
for line in relevant_docs_file:
    claims.append(line)

# # testing
# claim_0 = claims[0]
# for pair in claim_0['predicted_sentences_ner']:
#     print("\n")
#     print(pair[0])
#     print(pair[1])
#     print(get_sentence(pair[0], pair[1]))

STOP = -1
with jsonlines.open(relevant_sent_file, mode='w') as writer_c:
    for claim in claims:
        # get all possible sentences
        pair_sent_pair = {}

        for pair in claim['predicted_sentences_ner']:
            sentence = get_sentence(pair[0], pair[1])
            sentence = clean_sentence(sentence)
            title = pair[0].replace("_", " ")

            # if not title.lower() in sentence.lower():
            #     sentence = pair[0] + " " + sentence
            pair_sent_pair[sentence] = (pair[0], pair[1])

        for pair in claim['predicted_sentences']:
            sentence = get_sentence(pair[0], pair[1])
            sentence = clean_sentence(sentence)
            pair_sent_pair[sentence] = (pair[0], pair[1])

        corpus = []
        sentence_identifier = []
        for key in pair_sent_pair:
            corpus.append(key)
            sentence_identifier.append(pair_sent_pair[key])

        claim['predicted_sentences_bert'] = []

        # create embeddings
        corpus_embeddings = embedder.encode(corpus)
        query_embeddings = embedder.encode(claim['claim'])

        # get the n most similar sentences
        closest_n = 5
        for query, query_embedding in zip([claim['claim']], query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
            # print(query_embedding)
            # print(corpus_embeddings)
            # print(distances)
            # print(scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine"))
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for idx, distance in results[0:closest_n]:
                print(corpus[idx].strip(), "(Score: %.4f)" % (1 - distance))
                print(sentence_identifier[idx])
                # test_1 = query_embedding.reshape(-1, 1)
                # test_2 = [query_embedding]
                # test_3 = corpus_embeddings[idx]
                # test_4 = corpus_embeddings[idx].reshape(-1, 1)
                # print(scipy.spatial.distance.cdist([query_embedding],
                #                                    [corpus_embeddings[idx]],
                #                                    "cosine")[0])
                # print(cosine_similarity([query_embedding], [corpus_embeddings[idx]]))

                claim['predicted_sentences_bert'].append(sentence_identifier[idx])
        writer_c.write(claim)
        print(STOP)
        if STOP == 0:
            break
        else:
            STOP -= 1
