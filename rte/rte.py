import os
import codecs
import pickle
import json
import numpy as np
import subprocess

# global variable definition
labelToString = {
    0: "NOT ENOUGH INFO",
    1: 'SUPPORTS',
    2: 'REFUTES'}

def createTestSet(claim, candidateEvidences):
    
    testSetFile= codecs.open("./testSet_rte.jsonl", mode= "w", encoding= "utf-8")
    
    for elem in candidateEvidences:
        json.dump({"hypothesis": claim, "premise": elem}, testSetFile)
        testSetFile.write("\n")
    
    testSetFile.close()

def getPredictions():
    
    # call allennlp predictions shell script
    subprocess.call(['./allennlp_predictions.sh'])
    
    predsFile= codecs.open("./predictions_rte.json", mode= "r", encoding= "utf-8")
    
    rtePreds= []
    
    for line in predsFile.readlines():
        rtePreds.append(json.loads(line))
    
    predsFile.close()
    
    # for each element returns a dictionary of the form: {"predictedLabel": A, "confidence": B}
    predictionsProbability= []
    
    for prediction in rtePreds:
        maxIndex= np.argmax(np.asarray(prediction["label_probs"]))
        predictionsProbability.append({"predictedLabel": maxIndex, "confidence": prediction["label_probs"][maxIndex]})
    
    return predictionsProbability

def determinePredictedLabel(preds):
    
    nonePredictions = [elemIndex for elemIndex in range(len(preds)) if preds[elemIndex]["predictedLabel"] == 0]
    supportPredictions= [elemIndex for elemIndex in range(len(preds)) if preds[elemIndex]["predictedLabel"] == 1]
    contradictionPredictions= [elemIndex for elemIndex in range(len(preds)) if preds[elemIndex]["predictedLabel"] == 2]
    
    # determine the number of predictions for each case
    # return the prediction for the most predicted label and corresponding evidences
    numberOfPredictionsPerLabel= np.asarray([len(nonePredictions), len(supportPredictions), len(contradictionPredictions)])
    mostCommonPrediction= np.argmax(numberOfPredictionsPerLabel)
    
    if mostCommonPrediction == 0:
        return (0, [])
    elif mostCommonPredictions == 1:
        return (1, supportPredictions)
    else:
        return (2, contradictionPredictions)
    
    

def textual_entailment_evidence_retriever(claim, potential_evidence_sentences):
    
    createTestSet(claim, potential_evidence_sentences)
    
    preds= getPredictions()
    
    predictedLabel, evidencesIndexes = determinePredictedLabel(preds)
    
    return {"claim": claim, "label": labelToString[predictedLabel], "evidence": np.asarray(potential_evidence_sentences)[evidencesIndexes]}


"""
#### test #####

claim= "Gil was born in Porto"
candidateEvidences= ["Gil lives in Porto", "Gil was born in 1993 in Paris", "This document indicates that Gil was not born in Portugal"]

print(textual_entailment_evidence_retriever(claim, candidateEvidences))
"""
