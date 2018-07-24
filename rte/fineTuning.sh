#!/bin/bash
python -m allennlp.run fine-tune --model-archive ./snli_output/model.tar.gz --config-file ./fineTuning_config/decomposable_attention_fineTuning.json  --serialization-dir ./fever_output/
#TODO: check if "--overrides" parameter is relevant
