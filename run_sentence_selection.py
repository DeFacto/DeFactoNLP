import jsonlines
import codecs
import json
from sentence_transformers import SentenceTransformer
import scipy.spatial

wiki_split_docs_dir = "data/wiki-pages-split"
relevant_docs_file = "data/dev_concatenation.jsonl"
relevant_sent_file = "data/dev_sentence_selection.jsonl"


def get_sentence(doc, line_num):
    file = codecs.open(wiki_split_docs_dir + "/" + doc + ".json", "r", "utf-8")
    file = json.load(file)
    full_lines = file["lines"]
    lines = []
    for line in full_lines:
        lines.append(line['content'])
    sentence = lines[line_num]
    return sentence


# model = SentenceTransformer('bert-base-nli-mean-tokens')
embedder = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')

claims = []
for line in relevant_docs_file:
    claims.append(line)

# testing
claim_0 = claims[0]
for pair in claim_0['predicted_sentences_ner']:
    print(get_sentence(pair[0], pair[1]))

with jsonlines.open(relevant_sent_file, mode='w') as writer_c:
    corpus = []
    for claim in claims:
        # get all possible sentences
        for pair in claim['predicted_sentences_ner']:
            sentence = get_sentence(pair[0], pair[1])
            corpus.append(sentence)

        # create embeddings
        corpus_embeddings = embedder.encode(corpus)
        query_embeddings = embedder.encode(claim['claim'])

        # get the n most similar sentences
        closest_n = 5
        for query, query_embedding in zip(claim, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for idx, distance in results[0:closest_n]:
                print(corpus[idx].strip(), "(Score: %.4f)" % (1 - distance))
