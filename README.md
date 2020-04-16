# DeFactoNLP

DeFactoNLP is an automated fact-checking system designed for the FEVER 2018 Shared Task which was held at EMNLP 2018. It is capable of verifying claims and retrieving sentences from Wikipedia which support the assessment. This is accomplished through the usage of named entity recognition, TF-IDF vector comparison, and decomposable attention models.

### Achievements 
* 5th place in F1 Evidence Score
* 12th place in the FEVER Score 
### Cite
If you use this code, please cite:

```
@inproceedings{Reddy2018, 
title={DeFactoNLP: Fact Verification using Entity Recognition, TFIDF Vector Comparison and Decomposable Attention}, 
publisher={FEVER 2018, organised under EMNLP 2018}, 
author={Reddy, Aniketh Janardhan and Rocha, Gil and Esteves, Diego},
year={2018}
}
```

# System Structure 

![System Structure](https://github.com/DeFacto/DeFactoNLP/images/work_structure.png)

# Reproducing

To reproduce this work, an understanding of the files and scripts are needed.

### Data

All the files with the Claim information are in the [Data](/data) folder.
The files [train.jsonl](/data/train.jsonl), [dev.jsonl](/data/dev.jsonl) and [test.jsonl](/data/test.jsonl) are the files extracted from the FEVER database.  

The Wikipedia corpus can be downloaded running the script [download-raw-wiki.sh](/fever-baselines/scripts/download-raw-wiki.sh). To accelerate the algorithms, we divided every article into files using the script [split_wiki_into_indv_docs.py](split_wiki_into_indv_docs.py).

We also created a train subsample using the script [subsample_training_data.py](subsample_training_data.py).

The files [subsample_train_relevant_docs.jsonl](/data/subsample_train_relevant_docs.jsonl), [shared_task_dev_public_relevant_docs.jsonl](/data/shared_task_dev_public_relevant_docs.jsonl) and [shared_task_test_relevant_docs.jsonl](/data/shared_task_test_relevant_docs.jsonl) contain the information from the TF-IDF part of Document Retrieval (*predicted_pages*) and Sentence Retrieval (*predicted_sentences*). 

### TF-IDF (Document and Sentence Retrieval)

The TF-IDF results can be reproduced by running certain scripts inside [fever-baselines](/fever-baselines/) folder. First, download the [database](/fever-baselines#data-preparation) and than, run the [tf-idf](/fever-baselines#evidence-retrieval-evaluation) part. 
The files are already generated and can be found in the [data](/data) folder.

### NER and Entailment (Document and Sentence Retrieval)

The NER results and calculating all the sentence probability of supporting, refuting and being uninformative to the claim are calculated in the same script [predict.py](predict.py).
The scripts predict will load the TF-IDF results, calculate the NER documents and concatenate the predicted sentences from both. Further, those sentences will go to the RTE module and all probabilities will be calculated.
The functions of the NER Document Retrieval part are in [doc_retrieval.py](doc_retrieval.py).
The functions for the RTE module are in the folder [rte](/rte) that has their READ.me
The [predict.py](predict.py) script will generate a file for every claim with every calculated probability for each sentence.

### Label Classification

The Label Classification is performed training a Random Forest model. The [train_label_classifier.py](train_label_classifier.py) script will train and also predict the claim label based on the probabilities from the RTE model. 
The folder [entailment_predictions_train](/rte/entailment_predictions_train/) contains already calculated probabilities for our [subsample_train.jsonl](/data/subsample_train.jsonl). A file is genereted ready to be submitted.

