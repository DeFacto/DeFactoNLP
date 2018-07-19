#!/bin/bash

python -m allennlp.run predict ./snli_output/model.tar.gz  ./snli_data/testSet_rte.jsonl --output-file predictions_rte.json --silent
