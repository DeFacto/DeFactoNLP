#!/bin/bash

python -m allennlp.run predict ./snli_output/model.tar.gz  testSet_rte.jsonl --output-file predictions_rte_individual.json --silent