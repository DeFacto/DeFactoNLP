# DeFactoNLP

DeFactoNLP is an automated fact-checking system designed for the FEVER 2018 Shared Task which was held at EMNLP 2018. It is capable of verifying claims and retrieving sentences from Wikipedia which support the assessment. This is accomplished through the usage of named entity recognition, TF-IDF vector comparison and decomposable attention models.

If you use this code, please cite:

```
@inproceedings{Reddy2018, 
title={DeFactoNLP: Fact Verification using Entity Recognition, TFIDF Vector Comparison and Decomposable Attention}, 
publisher={FEVER 2018, organised under EMNLP 2018}, 
author={Reddy, Aniketh Janardhan and Rocha, Gil and Esteves, Diego},
year={2018}
}
```

##RUN:
1. Download Fever Data in fever-baselines/scripts/  
 run script: _download-raw-wiki.sh_  
 move to folder wiki-pages inside the data file that is already created.
 
2. Run split_wiki_indv_docs.py in order to increase search time. 
This will read the raw that and create a file for each article.

number of empty articles    = 20431  
number of files             = 5396106  
number of lines             = 42041604  
number of entities          = 167143495  
number of articles w/out id = 11  
