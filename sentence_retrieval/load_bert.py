import datetime
import time
import jsonlines
import torch
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from transformers import BertModel, BertTokenizer, AdamW, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import get_linear_schedule_with_warmup

os.chdir("..")

from run_sentence_selection import get_sentence, clean_sentence
from triple_sentence_selection import get_pairs_from_doc


class ClassForBert(nn.Module):

    def __init__(self, _model, hidden_size=0):
        super().__init__()
        self.model = _model
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, token_type_ids=''):
        _outputs = self.model(x)
        _outputs = self.dropout(_outputs[0][:, 0, :])
        _outputs = self.linear(_outputs)
        return self.sigmoid(_outputs)


PATH_MODEL = "output/bert_tuned.pth"
PATH_DATASET = "sentence_retrieval/train_sentence_retrieval.jsonl"
TRAIN = False
batch_size = 8
max_seq_length = 128  # max 512


def read_dataset():
    if not os.path.exists(PATH_DATASET):
        print("Dataset PATH does not exist")
        exit(-1)
    _dataset = jsonlines.open(PATH_DATASET)
    lines = []
    for line in _dataset:
        lines.append(line)
    return lines


def format_time(_elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((_elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Loading Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2,  # The number of output labels--2 for binary classification.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
print(model)
# exit(0)
# model = BertModel.from_pretrained('bert-base-uncased')
# print(model.config.hidden_size)
# model = ClassForBert(model, model.config.hidden_size)

if os.path.exists(PATH_MODEL):
    print("---- Model Loaded ----")
    model.load_state_dict(torch.load(PATH_MODEL))

if TRAIN:
    # Reading Dataset
    dataset = read_dataset()
    random.shuffle(dataset)
    x_train = dataset

    print(len(x_train))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Using pytorch device: {}".format(device))

    # Tokenizing Input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    inputs = []
    attention_masks = []
    scores = []
    x_train = x_train[:10]
    print(len(x_train))
    for pair in x_train:
        try:
            _input = '[CLS] ' + str(pair['sentence_a']) + ' [SEP] ' + str(pair['sentence_b']) + ' [SEP]'
            encoded_dict = tokenizer.encode_plus(_input,
                                                 add_special_tokens=False,
                                                 max_length=max_seq_length,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
            inputs.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            scores.append(int(pair['stance']))
        except Exception as e:
            print(e)
            print(pair)
            exit(0)
    # print(inputs)

    # Converting Dataset
    inputs = torch.cat(inputs, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    scores = torch.tensor(scores)
    x_train = TensorDataset(inputs, attention_masks, scores)

    train_size = int(0.9 * len(x_train))
    val_size = len(x_train) - train_size

    train_dataset, val_dataset = random_split(x_train, [train_size, val_size])

    # Create dataset
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The training samples.
        sampler=SequentialSampler(val_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0

        model.train()

        # For each batch of training data...

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains two pytorch tensors:
            #   [0]: input ids
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 labels=b_labels)

            total_train_loss += loss.item()
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    print("Saving Model")
    torch.save(model.state_dict(), PATH_MODEL)

    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    import seaborn as sns

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def get_bert_confidence(_input):
    _encoded_dict = tokenizer.encode_plus(_input,
                                          add_special_tokens=False,
                                          max_length=max_seq_length,
                                          pad_to_max_length=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
    # print(_encoded_dict['input_ids'])
    _input = _encoded_dict['input_ids']
    _mask = _encoded_dict['attention_mask']
    with torch.no_grad():
        _outputs = model(_input, _mask)
        # print(_outputs)
        predictions = _outputs[0]

        p = torch.nn.functional.softmax(predictions, dim=1)
        # print(p[0][1])
        # to calculate loss using probabilities you can do below
        # loss = torch.nn.functional.nll_loss(torch.log(p), y)
        # print(loss)
        # pred = np.argmax(predictions, axis=1).flatten()
        # print(float(predictions[0][1]))
        return float(p[0][1]), p


if not TRAIN:
    model.eval()

    to_load = "data/dev_test_transformers_baby.jsonl"
    to_write = "data/dev_test_pointwise_bert.jsonl"

    to_load = jsonlines.open(to_load)
    claims = []
    for line in to_load:
        claims.append(line)

    STOP = -1
    with jsonlines.open(to_write, mode='w') as writer_c:
        for claim in claims:
            # get all possible sentences
            pair_sent_pair = {}
            for doc in claim['predicted_pages_oie']:
                pairs = get_pairs_from_doc(doc)
                pairs += claim['predicted_sentences']
                for pair in pairs:
                    sentence = get_sentence(pair[0], pair[1])
                    sentence = sentence + " " + pair[0]
                    pair_sent_pair[sentence] = (pair[0], pair[1])

            corpus = []
            tuples = []
            for sentence in pair_sent_pair:
                pair = pair_sent_pair[sentence]
                sentence = "[CLS] " + claim['claim'] + " [SEP] " + sentence + " [SEP]"
                confidence, teste = get_bert_confidence(sentence)
                confidence = abs(confidence)
                tuples.append((pair[0], pair[1], confidence, sentence, teste))
            tuples.sort(key=lambda tup: tup[2], reverse=True)

            # print(tuples)
            pairs = []
            for tuple in tuples:
                print(tuple)
                pairs.append((tuple[0], tuple[1]))
            claim['predicted_sentences_bert_5'] = pairs[0:5]
            claim['predicted_sentences_bert_10'] = pairs[0:10]

            writer_c.write(claim)
            STOP -= 1
            if STOP == 0:
                break

    text_1 = "[CLS] bla bla bla whisky [SEP] cat [SEP]"
    text_2 = "[CLS] My dog is cute [SEP] My cat is ugly. [SEP]"
    text_3 = "[CLS] My dog is cute [SEP] WHAT ARE THOSE????????? [SEP]"

    print(get_bert_confidence(text_1))

    tokenized_text = tokenizer.tokenize(text_2)
    input_ids = torch.tensor(tokenizer.encode(tokenized_text,
                                              add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    # cls_token = last_hidden_states[:, 0, :]
    print(last_hidden_states)

    tokenized_text = tokenizer.tokenize(text_3)
    input_ids = torch.tensor(tokenizer.encode(tokenized_text,
                                              add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    # cls_token = last_hidden_states[:, 0, :]
    print(last_hidden_states)
