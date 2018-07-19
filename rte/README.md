# RTE Model 

## Installation requirements:
```
 > conda create -n allennlp python=3.6 anaconda
 > source activate allennlp
 > pip install allennlp
```

## How to run the code

### SNLI data only

#### Training:
```
bash train.sh
```
- input data is under: `/snli_input`
- output data (incl. trained model) will be placed under: `/snli_output`

#### Predictions:
```
bash predictions.sh
```
- predictions files will be place in the `/rte` directory in the filename `predictions_rte.json`

### Fine tune pretrained SNLI model with FEVER data

#### Training:
```
bash fineTuning.sh
```
- input data is under: `/fever_input`
- it uses the pretrained model saved at: `/snli_output/model.tar.gz`
- output data (incl. trained model) will be place under: `/fever_output`

#### Predictions:
```
bash predictions_fineTune.sh
```
- predictions files will be place in the `/rte` directory in the filename `predictions_rte_fever.json`
