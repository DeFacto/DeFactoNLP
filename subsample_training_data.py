import os
import jsonlines
from random import shuffle

train_file = "data/train.jsonl"

train_file = jsonlines.open(train_file)
train_set = []

support = []
refute = []
notenoughinfo = []
num_required_support = 3000
num_required_refute = 3000
num_required_notenoughinfo = 4000

for lines in train_file:
	train_set.append(lines)

shuffle(train_set)

for lines in train_set:
	if lines['label'] == "SUPPORTS" and num_required_support>0:
		support.append(lines)
		num_required_support = num_required_support - 1
	elif lines['label'] == "REFUTES" and num_required_refute>0:
		refute.append(lines)
		num_required_refute = num_required_refute - 1
	elif lines['label'] == "NOT ENOUGH INFO" and num_required_notenoughinfo>0:
		notenoughinfo.append(lines)
		num_required_notenoughinfo = num_required_notenoughinfo - 1

subsampled = support + refute + notenoughinfo
shuffle(subsampled)

print(len(train_set))
print(len(subsampled))

subsampled_file = "data/subsample_train.jsonl"
with jsonlines.open(subsampled_file, mode='w') as writer:
	for s in subsampled:
		writer.write(s)

