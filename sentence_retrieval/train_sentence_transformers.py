import logging
import math
import jsonlines
import os

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from random import shuffle

os.chdir('..')



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

train_data_file = "sentence_retrieval/train_sentence_retrieval.jsonl"
train_data_file = jsonlines.open(train_data_file)
train_data = []
for line in train_data_file:
    train_data.append(line)

model_name = 'bert-base-nli-mean-tokens'
batch_size = 16
num_epochs = 1
model_save_path = 'output/subsample_train-' \
                  + model_name + '-' + datetime.now().strftime("%Y-%m ""-%d_%H-%M-%S")

model = SentenceTransformer(model_name)
shuffle(train_data)

logging.info("Reading Subsample of Train and Dev Dataset")
examples_train = []
examples_dev = []
_id = 0
for line in train_data:
    _id += 1
    sentence_a = line['sentence_a']
    sentence_b = line['sentence_b']
    stance = line['stance']
    if _id % 50 == 0:
        examples_dev.append(InputExample(str(_id),
                                         texts=[sentence_a, sentence_b],
                                         label=float(stance)))
    examples_train.append(InputExample(str(_id),
                                       texts=[sentence_a, sentence_b],
                                       label=float(stance)))

print(len(examples_dev))
logging.info("Train Data Loaded")
train_data = SentencesDataset(examples_train, model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Dev Data Loaded")
dev_data = SentencesDataset(examples_dev, model)
dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data) * num_epochs / batch_size * 0.05)  # 5% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

