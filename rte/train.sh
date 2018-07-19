#!/bin/bash
python -m allennlp.run train training_config/decomposable_attention.json -s ./snli_output
