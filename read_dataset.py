import jsonlines
import doc_retrieval
import sentence_retrieval

train_file = "data/train.jsonl"
test_file = "data/test.jsonl"
dev_file = "data/dev.jsonl"

wiki_dir = 'data/wiki-pages/wiki-pages'
wiki_split_docs_dir = "data/wiki-pages/wiki-pages-split"

train_file = jsonlines.open(train_file)
test_file = jsonlines.open(test_file)
dev_file = jsonlines.open(dev_file)

train_set = []
test_set = []
dev_set = []

for lines in train_file:
	train_set.append(lines)
# for lines in test_file:
# 	test_set.append(lines)
# for lines in dev_file:
# 	dev_set.append(lines)

total = 0.0
retrieved = 0.0

for example in train_set[:5]:
	relevant_docs = doc_retrieval.getRelevantDocs(example['claim'])
	print(relevant_docs)
	relevant_sentences = sentence_retrieval.getRelevantSentences(relevant_docs,wiki_split_docs_dir)
	print(relevant_sentences)
# 	if example['label']=="REFUTES" or example['label']=="SUPPORTS":
# 		for actual_evidence in example['evidence'][0]:
# 			total+=1.0
# 			if actual_evidence[2] in relevant_docs:
# 				retrieved += 1.0
# 				# print actual_evidence[2]

# print total
# print retrieved
# print retrieved/total