# -*- coding: utf-8 -*-
# Calculate PEA score on a given dataset 

# Import libraries ----
from collections import Counter
import pandas as pd
import numpy as np


# Emotion wheel to radian ----
emotion_wheel = {
    'anger': 180,
    'anticipation': 135,
    'joy': 90,
    'trust': 45,
    'fear': 0,
    'surprise': 315,
    'sadness': 270,
    'disgust': 225,
    'none': 360,
}

rev_emotion_wheel = {
        180: 'anger',
        135: 'anticipation',
        90: 'joy',
        45: 'trust',
        0: 'fear',
        315: 'surprise',
        270: 'sadness',
        225: 'disgust',
        360: 'none',
}


def filter_na(df): 
    '''
    Filter out tweets which have more than 60% 'None of the above' annotation
    - df: data frame to apply this function to  
    '''
    occur = []
    for item in df['Annotation']: 
        occur.append(Counter(x for sublist in item for x in sublist)['none']/len(item))
    df['NA occurrence'] = occur
    df = df[df['NA occurrence'] < 0.6].reset_index(drop = True)
    del df['NA occurrence']
    return df


def convert_radian(df_col): 
    '''
    Convert emotions to radian values
    - df_col: column of the dataframe to apply this function to 
    '''
    for asm in df_col: 
        for key in list(asm.keys()):
            for i in range(len(asm[key])): 
                asm[key][i] = emotion_wheel[str(asm[key][i])]
    return df_col


def pairwise_calc(emo1, assignment2): 
    '''
    Calculate the greatest distance between an emotion (assignment 1) and annotations of assignment 2----
    - emo1: an annotation within assignment 1
    - assignment2: annotation set of assignment 2
    '''
    dists = []
    for emo2 in assignment2: 
        calc = abs(emo2 - emo1)
        if calc >= 180:
            calc = 360 - calc 
        # Normalize
        calc /= 180.  
        dists.append(1. - calc)
    return max(dists)


def add_dict(agreement_score, assignment, score): 
    '''
    Add score to agreement_score dictionary
    - agreement_score: dictionary to store assignment ID and scores
    - assignment: assignment ID
    - score: pairwise agreement score
    '''
    if assignment not in agreement_score: 
        agreement_score[assignment] = [score]
    else: 
        agreement_score.get(assignment).append(score)
    return agreement_score


def calculate_pea(df_col): 
    '''
    Calculate PEA score for each tweet
    - df_col: column of the dataframe to apply this function to 
    '''
    agreement_score = {}
    pea = {}
    for item in df_col: 
        if len(item) == 1: 
            agreement_score = add_dict(agreement_score, list(item.keys())[0], 1.0)
        else: 
            for assignment1 in list(item): 
                for assignment2 in list(item): 
                    if assignment1 != assignment2: 
                        total_agg = 0
                        for emo1 in item[assignment1]: 
                            total_agg += pairwise_calc(emo1, item[assignment2])
                        agreement_score = add_dict(agreement_score, assignment1, total_agg/len(item[assignment1]))
    
    for key, value in agreement_score.items(): 
        pea[key] = sum(value)/len(value)
    return pea


def calculate_threshold(df_col, percentile): 
    '''
    Determine assignments under the threshold 
    - df_col: column of the dataframe to apply this function to 
    - percentile: 95th or 80th
    return IDs for assignments under threshold
    '''
    rm_asmID = {}
    if type(df_col) == type(pd.Series([])): 
        agreement_val = np.asarray(df_col)
    else: 
        agreement_val = np.asarray(list(df_col.values()))
    threshold = np.percentile(agreement_val, 100-percentile)
    print("The threshold is", str(threshold))

    # Determine assignment ID that are under threshold
    for key, value in df_col.items(): 
        if value <= threshold: 
            rm_asmID[key] = value
    rm_asmID = sorted(rm_asmID.items(), key=lambda x: x[1])
    return threshold, [asm[0] for asm in rm_asmID]

