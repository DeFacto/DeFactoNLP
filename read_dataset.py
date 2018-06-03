import jsonlines
import doc_retrieval

train_file = "data/train.jsonl"
test_file = "data/test.jsonl"
dev_file = "data/dev.jsonl"

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

for example in train_set[:10]:
	relevant_docs = doc_retrieval.getRelevantDocs(example['claim'])