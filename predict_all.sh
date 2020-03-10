#!/bin/bash
mkdir before_predictions
mkdir concatenations
echo "Predict Train"
python predict_old.py "data/subsample_train_relevant_docs.jsonl" "before_predictions/predictions_train.jsonl" "concatenations/subsample_train_concatenation.jsonl"
echo "Predict Dev"
python predict_old.py "data/shared_task_dev_public_relevant_docs.jsonl" "before_predictions/predictions_dev.jsonl" "concatenations/dev_concatenation.jsonl"
echo "Predict Test"
python predict_old.pt "data/subsample_train_relevant_docs.jsonl" "before_predictions/predictions_train.jsonl" "concatenations/subsample_train_concatenation.jsonl"