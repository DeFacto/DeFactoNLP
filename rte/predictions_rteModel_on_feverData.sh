#!/bin/bash

python -m allennlp.run predict snli_output/model.tar.gz  ./fever_data/test_fever_snliFormat.jsonl --output-file predictions_rteModel_on_feverData.json --silent
