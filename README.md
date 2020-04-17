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
* **INCLUDE_SENTENCE_BERT** --> if the input file contains ***Sentence-Transformers** Predicted SENTENCES* and you want to include them as relevant sentences
* **RUN_DOC_TRIPLE_BASED** --> to *Predict **Triple Based** Relevant DOCUMENTS* 
* **RUN_SENT_TRIPLE_BASED** --> to *Predict **Triple Based** Relevant SENTENCES* 
* **RUN_RTE** -> to run **Recognising Textual Entailment** to calculate the probabilities for every *Relevant Sentences*

Changing these variables will allow to run every step as required, making possible to run every step in a separate way, all at the same time or even include other Retrieval techniques using files with that information.

To generate the final predictions, run [Label Classification](#Label-Classification)

You can run all metrics using the script [metrics.py](metrics.py).

# In-Depth Information

Always check files paths before running anything

### Data

All the files with the Claim information are in the [Data](/data) folder.
The files [train.jsonl](/data/train.jsonl), [dev.jsonl](/data/dev.jsonl) and [test.jsonl](/data/test.jsonl) are the files extracted from the FEVER database.  

The Wikipedia corpus can be downloaded running the script [download-raw-wiki.sh](/fever-baselines/scripts/download-raw-wiki.sh). To accelerate the algorithms, we divided every article into files using the script [split_wiki_into_indv_docs.py](split_wiki_into_indv_docs.py).

We also created a train subsample using the script [subsample_training_data.py](subsample_training_data.py).

The files [subsample_train_relevant_docs.jsonl](/data/subsample_train_relevant_docs.jsonl), [shared_task_dev_public_relevant_docs.jsonl](/data/shared_task_dev_public_relevant_docs.jsonl) and [shared_task_test_relevant_docs.jsonl](/data/shared_task_test_relevant_docs.jsonl) contain the information from the TF-IDF part of Document Retrieval (*predicted_pages*) and Sentence Retrieval (*predicted_sentences*). 

All the files have certain keywords. OIE stands for Open Information Extraction (in Document Retrieval). SENTENCE was performed a Triple-Based method for Sentence Selection. Important to verify the first line of every file to know what Retrieval Method was made.

### TF-IDF (Document and Sentence Retrieval)

The TF-IDF results can be reproduced by running certain scripts inside [fever-baselines](/fever-baselines/) folder. First, download the [database](/fever-baselines#data-preparation) and than, run the [tf-idf](/fever-baselines#evidence-retrieval-evaluation) part. 
The files are already generated and can be found in the [data](/data) folder.

### Label Classification

The Label Classification is performed training a Random Forest model. The [train_label_classifier.py](train_label_classifier.py) script will train and also predict the claim label based on the probabilities from the RTE model. 
The folder [entailment_predictions_train](/rte/entailment_predictions_train/) contains already calculated probabilities for our [subsample_train.jsonl](/data/subsample_train.jsonl). A file is genereted ready to be submitted.

### Metrics

Running [metrics.py](metrics.py) will give you stats in detail about each of the three tasks. 

### Triple Based Model (Sentence Retrieval)

Running [proof_extraction_train.py](proof_extraction_train.py) will train the model. You need to give as argument number 0 to create the dataset (relevant and non-relevant sentences, number 1 to extract all features from the sentences and number 2 to train the model. Ideally, use the three numbers as arguments.

### Word2vec

For Document Retrieval we tried to use Word2vec model to extract the nearby documents to a given claim. Although promising, it didn't get the promising results. We think that is due to the lack of better training. 
There are two comparisons:
  1. between the title of the document and the sentence
  2. between the title of the document and every word of the claim (without stopwords) 
The main issue is how slow the process is. Indexing the titles would improve the processing speed. Number 2 is more promising, although even slower. You can find the code in [word2vec.py](word2vec.py).

### Doc2vec

For Document Retrieval we tried to use Doc2vec model to extract the nearby documents to a given claim. You can find the code in [doc2vec.py](doc2vec.py) although it didn't give any promising results since the generated vectors for the claims and the Documents are very different. 

### NER and Triple Based (Document Retrieval)

The file [doc_retrieval.py](doc_retrieval.py) contains the information that was used to find the most relevant Documents using NER and also our Triple Based approach.

### Sentence-Transformers

The code to run Sentence-Transformers is found [run_sentence_selection.py](run_sentence_selection.py). This will choose the top5 most similar sentences while [run_sentence_selection_doc.py](run_sentence_selection_doc.py) will choose the top2 sentences for every Retrieved Document.
Our fine-tuning model script code is in [train_sentence_model.py](train_sentence_model.py).
