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

The system is based on three major tasks (Document Retrieval, Sentence Retrieval, Label Classification). Each task was performed using different techniques:

* Document Retrieval
  * TF-IDF 
  * NER
  * Triple-Based
* Sentence Retrieval
  * TF-IDF
  * Triple-Based Model
  * [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
* Label Classification
  * RTE Model + Random Forest model

# Run

You can run Document Retrieval and Sentence Retrieval by running the following script: [generate_rte_preds.py](/generate_rte_preds.py).

The script contains the 6 boolean variables:
* **INCLUDE_NER** --> if the input file contains ***NER** Predicted DOCUMENTS* and you want to include them as relevant documents
* **INCLUDE_TRIPLE_BASED** --> if the input file contains ***Triple Based** Predicted DOCUMENTS* and you want to include them as relevant documents
* **INCLUDE_SENTENCE_BERT** --> if the input file contains ***Triples Based** Predicted SENTENCES* and you want to include them as relevant sentences
* **RUN_DOC_TRIPLE_BASED** --> to *Predict **Triple Based** Relevant DOCUMENTS* 
* **RUN_SENT_TRIPLE_BASED** --> to *Predict **Triple Based** Relevant SENTENCES* 
* **RUN_RTE** -> to run **Recognising Textual Entailment** to calculate the probabilities for every *Relevant Sentences*

Changing this variables will allow to run every step as required, making possible to run every step in a seperate way, all at the same time or even include other Retrieval techniques using files with that information.

To generate the final predictions, run [Label Classification](#Label Classification)

### Data

All the files with the Claim information are in the [Data](/data) folder.
The files [train.jsonl](/data/train.jsonl), [dev.jsonl](/data/dev.jsonl) and [test.jsonl](/data/test.jsonl) are the files extracted from the FEVER database.  

The Wikipedia corpus can be downloaded running the script [download-raw-wiki.sh](/fever-baselines/scripts/download-raw-wiki.sh). To accelerate the algorithms, we divided every article into files using the script [split_wiki_into_indv_docs.py](split_wiki_into_indv_docs.py).

We also created a train subsample using the script [subsample_training_data.py](subsample_training_data.py).

The files [subsample_train_relevant_docs.jsonl](/data/subsample_train_relevant_docs.jsonl), [shared_task_dev_public_relevant_docs.jsonl](/data/shared_task_dev_public_relevant_docs.jsonl) and [shared_task_test_relevant_docs.jsonl](/data/shared_task_test_relevant_docs.jsonl) contain the information from the TF-IDF part of Document Retrieval (*predicted_pages*) and Sentence Retrieval (*predicted_sentences*). 

All the files have certain keyworks. OIE stands for Open Information Extraction (in Document Retrieval). SENTENCE was performed a Triple-Based method for Sentence Selection. Important to verify the first line of every file to know what Retrieval Method was made.

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

