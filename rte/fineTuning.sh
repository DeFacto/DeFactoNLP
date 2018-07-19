#!/bin/bash
python -m allennlp.run fine-tune --model-archive ./snli_output/model.tar.gz --config-file ./fineTuning_config/decomposable_attention_fineTuning.json  --serialization-dir ./fever_output/  --file-friendly-logging True
#TODO: check if "--overrides" parameter is relevant
