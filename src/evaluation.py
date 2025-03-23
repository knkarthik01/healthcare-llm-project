"""
Module for evaluating LLM responses to diabetes risk assessment prompts.
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def extract_risk_level(response: str) -> str:
    """
    Extract the predicted risk level from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        Extracted risk level (Low, Medium, High, or Unknown)
    """
    # Use regex to find risk assessment
    low_pattern = r'\b(low risk|low-risk|minimal risk)\b'
    medium_pattern = r'\b(medium risk|medium-risk|moderate risk)\b'
    high_pattern = r'\b(high risk|high-risk|elevated risk|significant risk)\b'
    
    # Check for phrases like "I assess this as high risk" or "The patient has a medium risk"
    if re.search(low_pattern, response.lower()):
        return 'Low'
    elif re.search(medium_pattern, response.lower()):
        return 'Medium'
    elif re.search(high_pattern, response.lower()):
        return 'High'
    
    # Check for final assessment line
    final_assessment = re.search(r'final assessment:?\s*(.*)', response.lower())
    if final_assessment:
        assessment_text = final_assessment.group(1)
        if 'low' in assessment_text and not ('medium' in assessment_text or 'high' in assessment_text):
            return 'Low'
        elif 'medium' in assessment_text or 'moderate' in assessment_text:
            return 'Medium'
        elif 'high' in assessment_text:
            return 'High'
    
    return 'Unknown'

def count_factors_mentioned(response: str) -> int:
    """
    Count the number of diabetes risk factors mentioned in the response.
    
    Args:
        response: LLM response text
        
    Returns:
        Count of risk factors mentioned
    """
    # List of common diabetes risk factors
    factors = [
        'age', 'bmi', 'weight', 'obese', 'obesity', 'overweight',
        'glucose', 'blood sugar', 'hba1c', 'a1c', 'hemoglobin',
        'hypertension', 'blood pressure', 'family history',
        'prediabetes', 'gestational', 'ethnicity', 'race'
    ]
    
    # Count unique factors mentioned
    mentioned = set()
    for factor in factors:
        if factor in response.lower():
            mentioned.add(factor)
    
    return len(mentioned)

def evaluate_reasoning_depth(response: str) -> int:
    """
    Evaluate the depth of reasoning on a scale of 1-5.
    
    Args:
        response: LLM response text
        
    Returns:
        Score from 1-5 on reasoning depth
    """
    # Count sentences as a basic measure of response depth
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Count risk factors mentioned
    factor_count = count_factors_mentioned(response)
    
    # Check for structured reasoning steps
    has_structured_steps = False
    step_keywords = ['first', 'second', 'third', 'next', 'finally', 'step', 'factor']
    step_count = 0
    for keyword in step_keywords:
        step_count += len(re.findall(r'\b' + keyword + r'\b', response.lower()))
    has_structured_steps = step_count >= 3
    
    # Check for medical reasoning with specific thresholds
    has_medical_thresholds = any(threshold in response.lower() for threshold in 
                                ['140', '200', '5.7', '6.5', '100', '126', '30'])
    
    # Compute score
    score = 1  # Minimum score
    
    if sentence_count >= 3:
        score += 1
    if factor_count >= 3:
        score += 1
    if has_structured_steps:
        score += 1
    if has_medical_thresholds:
        score += 1
    
    return min(score, 5)  # Cap at 5

def evaluate_response(response: str, true_risk: str) -> Dict[str, Any]:
    """
    Evaluate a single LLM response.
    
    Args:
        response: LLM response text
        true_risk: Actual risk level (Low, Medium, High)
        
    Returns:
        Dictionary with evaluation metrics
    """
    predicted_risk = extract_risk_level(response)
    
    return {
        'true_risk': true_risk,
        'predicted_risk': predicted_risk,
        'correct': predicted_risk == true_risk,
        'reasoning_depth': evaluate_reasoning_depth(response),
        'factors_mentioned': count_factors_mentioned(response),
        'response_length': len(response)
    }

def aggregate_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate evaluation results across multiple patients.
    
    Args:
        evaluations: List of individual evaluations
        
    Returns:
        Dictionary with aggregated metrics
    """
    df = pd.DataFrame(evaluations)
    
    # Basic accuracy
    accuracy = df['correct'].mean()
    
    # Reasoning depth stats
    avg_reasoning_depth = df['reasoning_depth'].mean()
    
    # Factors mentioned stats
    avg_factors = df['factors_mentioned'].mean()
    
    # Get confusion matrix
    y_true = df['true_risk'].tolist()
    y_pred = df['predicted_risk'].tolist()
    
    # Filter out "Unknown" predictions if any
    valid_indices = [i for i, pred in enumerate(y_pred) if pred != 'Unknown']
    if valid_indices:
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]
    else:
        y_true_valid = y_true
        y_pred_valid = y_pred
    
    # Get unique classes while preserving order (Low, Medium, High)
    classes = ['Low', 'Medium', 'High']
    y_true_classes = sorted(set(y_true_valid), key=lambda x: classes.index(x) if x in classes else 999)
    
    # Compute confusion matrix if there's data to compute on
    cm = None
    precision = recall = f1 = None
    if len(y_true_valid) > 0 and len(set(y_true_valid)) > 1 and len(set(y_pred_valid)) > 1:
        try:
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_valid, y_pred_valid, 
                                                                       labels=classes, average=None)
        except:
            # Fall back if there are issues with the metrics computation
            pass
    
    return {
        'accuracy': accuracy,
        'avg_reasoning_depth': avg_reasoning_depth,
        'avg_factors_mentioned': avg_factors,
        'confusion_matrix': cm,
        'classes': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_evaluations': len(evaluations),
        'unknown_rate': (df['predicted_risk'] == 'Unknown').mean()
    }
