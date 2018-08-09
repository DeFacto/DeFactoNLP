#!/bin/bash

python -m allennlp.run predict ./fever_output/model.tar.gz  ./fever_data/test_fever_snliFormat.jsonl --output-file predictions_rte_fever_testSet.json --silent
