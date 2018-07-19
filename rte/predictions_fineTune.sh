#!/bin/bash

python -m allennlp.run predict ./fever_data/model.tar.gz  ./fever_data/test_fever_snliFormat.jsonl --output-file predictions_rte_fever.json --silent
