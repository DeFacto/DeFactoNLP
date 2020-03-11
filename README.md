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
## WARNINGS:
* Don't forget to check all *PATHS* in every script all the time
* Don't run the RTE model before training the Random Forest model.

##RUN:
1. Download Fever Data in fever-baselines/scripts/  
 run script: _download-raw-wiki.sh_  
 move to folder wiki-pages inside the data file that is already created.
 
2. Run split_wiki_indv_docs.py in order to increase search time. 
This will read the raw that and create a file for each article.

3. There is a need to setup at least one environment:
    * Inside the RTE folder there is a READ.ME and a requirements file.
    * The TF-IDF part of the article it comes within the fever-baselines. In that folder there 
    is a requirements file. It's needed to setup the fever database and run the required scripts. 
    * TF-IDF files are already available inside the folder data/ under the name "relevant_docs".
    * To reproduce the TF-IDF files the following script needs to run. It is found
    [here](https://github.com/DeFacto/DeFactoNLP/tree/master/fever-baselines#evidence-retrieval-evaluation)
    inside the READ.ME of the fever-baselines
   
4. Levenshtein part and concatenation is achieved by run the script: predict.py. It will generate
a file with all the documents and sentences and the concatenation part. The RTE model will also
creates a file for every claim. Each file contains the probabilites of the RTE prediction
 for every claim versus possible evidence.
 
5. To label the claims, the Random Forest model is created by run the script train_label_classifier.py . It will generate a file that 
is found in the folder predictions/.

6. You can run metrics.py to generate an evaluation of the entire pipeline.

## Train:

This work used a subsample of the training data using the script: subsample_training_data.py
  
In order to train the RTE model, all the explanations are in the specific READ.ME inside folder rte/
  
To train the Random Forest just run the script. It will also generate the predictions.
Comment what isn't needed.

### Some numbers:

number of empty articles    = 20431  
number of files             = 5396106  
number of lines             = 42041604  
number of entities          = 167143495  
number of articles w/out id = 11  

