from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader, InputExample
import logging
from datetime import datetime
import logging
import codecs
import json
import jsonlines
import tqdm

wiki_split_docs_dir = "../wiki-pages-split"


def get_sentence(doc, line_num):
    try:
        file = codecs.open(wiki_split_docs_dir + "/" + doc + ".json", "r", "utf-8")
    except:
        print("Failed Loading: " + str(doc))
        return "-1", ""

    file = json.load(file)
    full_lines = file["lines"]
    _lines = []
    for line in full_lines:
        _lines.append(line['content'])

    _non_related_sentences = set()
    sentence = ""
    for i in range(len(_lines)):
        if _lines[i] == "":
            # empty line...
            continue
        if i == line_num:
            sentence = _lines[line_num]
        else:
            _non_related_sentences.add(_lines[i])
    sentence_2 = _lines[line_num]
    if sentence != sentence_2:
        print("Sanity check failed!!!!!!!!!!!!!!!!!!!!!!")

    if len(_non_related_sentences):
        _non_related_sentences = list(_non_related_sentences)[0]
    else:
        _non_related_sentences = "ERROR404"
    #print(_non_related_sentences)
    #print(len(_non_related_sentences))
    return sentence, _non_related_sentences


def get_labels():
    # contradiction -> REFUTES # entailment -> SUPPORTS # neutral -> Not Enough Information
    return {"refutes": 0, "supports": 0, "neutral": 2}


def get_num_labels():
    return len(get_labels())


def map_label(_label):
    return get_labels()[_label.strip().lower()]


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

train_file = "data/subsample_train.jsonl"
train_file = jsonlines.open(train_file)
train_set = []
for lines in train_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    train_set.append(lines)

dev_file = "data/dev.jsonl"
dev_file = jsonlines.open(dev_file)
dev_set = []
for lines in dev_file:
    lines['claim'] = lines['claim'].replace("-LRB-", " ( ")
    lines['claim'] = lines['claim'].replace("-RRB-", " ) ")
    dev_set.append(lines)

model_name = 'bert-base-nli-mean-tokens'
batch_size = 16
num_epochs = 1
train_num_labels = get_num_labels()
model_save_path = 'output/subsample_train-' \
                  + model_name + '-' + datetime.now().strftime("%Y-%m ""-%d_%H-%M-%S")
# sts_reader = STSDataReader('datasets/stsbenchmark', normalize_scores=True)

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

STOP = -10

logging.info("Reading Subsample of Train Dataset")
examples_train = []
neutral = 0
non_neutral = 0
for example in train_set:
    sentence_a = example['claim']
    evidences = example['evidence']
    label = example['label']

    if label == "NOT ENOUGH INFO":
        continue

    for evidence in evidences:
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
            #all_non_related_sentences |= non_related_sentences
            examples_train.append(InputExample(example['id'], texts=[sentence_a, sentence_b], label=map_label(label)))
            print(sentence_b)
            print(non_related_sentences)
            if non_related_sentences != "ERROR404":
                examples_train.append(InputExample(example['id'],
                                               texts=[sentence_a, non_related_sentences],
                                               label=map_label("neutral")))
                neutral += 1
            non_neutral += 1
        all_non_related_sentences = list(all_non_related_sentences)
        for non_related_sentence in all_non_related_sentences:

            #print(non_related_sentence)
            if non_related_sentence != "" and False:
                print("UPSI")
                examples_train.append(InputExample(example['id'],
                                               texts=[sentence_a, non_related_sentence],
                                               label=map_label("neutral")))
    if STOP == 0:
        break
    else:
        STOP -= 1

print(non_neutral)
print(neutral)
logging.info("Train Data Loaded")
train_data = SentencesDataset(examples_train, model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model,
                                sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=train_num_labels)

logging.info("Reading Dev Dataset")
examples_dev = []
STOP = 500
for example in dev_set:
    sentence_a = example['claim']
    evidences = example['evidence']
    label = example['label']
    if label == "NOT ENOUGH INFO":
        continue
    for evidence in evidences:
        pairs = set()
        if len(evidence) > 1:
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
            #all_non_related_sentences |= non_related_sentences
            examples_dev.append(InputExample(example['id'], texts=[sentence_a, sentence_b], label=1))

        #for non_related_sentence in all_non_related_sentences:
            if non_related_sentences != "ERROR404":
                examples_dev.append(InputExample(example['id'],
                                             texts=[sentence_a, non_related_sentences],
                                             label=0))
    if STOP == 0:
        break
    else:
        STOP -= 1

logging.info("Dev Data Loaded")
dev_data = SentencesDataset(examples=examples_dev, model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
num_epochs = 3

warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1)  # 1 0% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )
