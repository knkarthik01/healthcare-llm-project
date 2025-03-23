"""
Module containing different prompting techniques for LLM-based diabetes risk assessment.
"""
from typing import List, Dict, Any

def basic_prompt(patient_text: str) -> str:
    """
    Generate a basic prompt for diabetes risk prediction.
    
    Args:
        patient_text: Text description of patient
        
    Returns:
        Prompt for LLM
    """
    return f"""
    Based on the following patient information, assess their risk of developing type 2 diabetes:
    
    {patient_text}
    
    Provide your assessment of whether this patient has a Low, Medium, or High risk of developing 
    type 2 diabetes and explain your reasoning.
    """

def in_context_learning_prompt(patient_text: str, examples: List[Dict[str, Any]]) -> str:
    """
    Generate an in-context learning prompt with examples.
    
    Args:
        patient_text: Text description of patient
        examples: List of example assessments to include in the prompt
        
    Returns:
        Prompt for LLM
    """
    prompt = "I'll show you examples of patient diabetes risk assessments, then ask you to assess a new patient.\n\n"
    
    # Add examples
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Patient information:\n{example['patient_text']}\n\n"
        prompt += f"Assessment: This patient has a {example['risk_level']} risk of developing type 2 diabetes because {example['reasoning']}\n\n"
    
    # Add target patient
    prompt += f"New patient to assess:\n{patient_text}\n\nProvide your assessment of whether this patient has a Low, Medium, or High risk of developing type 2 diabetes and explain your reasoning."
    return prompt

def few_shot_learning_prompt(patient_text: str, examples: List[Dict[str, Any]]) -> str:
    """
    Generate a few-shot learning prompt with diverse examples.
    
    Args:
        patient_text: Text description of patient
        examples: List of example assessments to include in the prompt
        
    Returns:
        Prompt for LLM
    """
    # Ensure we have examples covering different risk levels
    prompt = "I'll show you how to assess diabetes risk for patients with different risk levels, then ask you to assess a new patient.\n\n"
    
    # Group examples by risk level to make sure we cover all levels
    grouped_examples = {
        'Low': [ex for ex in examples if ex['risk_level'] == 'Low'],
        'Medium': [ex for ex in examples if ex['risk_level'] == 'Medium'],
        'High': [ex for ex in examples if ex['risk_level'] == 'High']
    }
    
    # Get one example from each risk level
    for risk_level in ['Low', 'Medium', 'High']:
        if grouped_examples[risk_level]:
            example = grouped_examples[risk_level][0]
            prompt += f"{risk_level} Risk Example:\n"
            prompt += f"Patient information:\n{example['patient_text']}\n\n"
            prompt += f"Assessment: This patient has a {risk_level} risk of developing type 2 diabetes because {example['reasoning']}\n\n"
    
    # Add target patient
    prompt += f"New patient to assess:\n{patient_text}\n\nProvide your assessment of whether this patient has a Low, Medium, or High risk of developing type 2 diabetes and explain your reasoning."
    return prompt

def chain_of_thought_prompt(patient_text: str) -> str:
    """
    Generate a chain-of-thought prompt for diabetes risk assessment.
    
    Args:
        patient_text: Text description of patient
        
    Returns:
        Prompt for LLM
    """
    return f"""
    Based on the following patient information, assess their risk of developing type 2 diabetes:
    
    {patient_text}
    
    Think step by step to determine the patient's risk level:
    
    1. First, identify the key risk factors for type 2 diabetes in this patient's profile
    2. For each risk factor, assess its severity and contribution to overall risk
    3. Consider how these risk factors interact with each other
    4. Determine if there are any protective factors that might reduce risk
    5. Based on the balance of risk and protective factors, determine the overall risk level (Low, Medium, or High)
    6. Provide specific recommendations based on the risk assessment
    
    After thinking through these steps, provide your final assessment of whether this patient has a Low, Medium, or High risk of developing type 2 diabetes.
    """

def tree_of_thought_prompt(patient_text: str) -> str:
    """
    Generate a tree-of-thought prompt that explores multiple possibilities.
    
    Args:
        patient_text: Text description of patient
        
    Returns:
        Prompt for LLM
    """
    return f"""
    Based on the following patient information, assess their risk of developing type 2 diabetes:
    
    {patient_text}
    
    Consider three possible assessments: Low Risk, Medium Risk, and High Risk.
    
    For each possibility:
    1. Low Risk:
       - Evidence supporting this assessment:
       - Evidence against this assessment:
       - Likelihood of this assessment being correct:
    
    2. Medium Risk:
       - Evidence supporting this assessment:
       - Evidence against this assessment:
       - Likelihood of this assessment being correct:
    
    3. High Risk:
       - Evidence supporting this assessment:
       - Evidence against this assessment:
       - Likelihood of this assessment being correct:
    
    After considering all possibilities, provide your final assessment of whether this patient has a Low, Medium, or High risk of developing type 2 diabetes, along with your reasoning.
    """

def combined_approach_prompt(patient_text: str, examples: List[Dict[str, Any]]) -> str:
    """
    Combine in-context learning with chain-of-thought reasoning.
    
    Args:
        patient_text: Text description of patient
        examples: List of example assessments to include in the prompt
        
    Returns:
        Prompt for LLM
    """
    prompt = "I'll show you examples of diabetes risk assessments with reasoning, then ask you to assess a new patient.\n\n"
    
    # Add examples with reasoning steps (one per risk level if available)
    for risk_level in ['Low', 'Medium', 'High']:
        matching_examples = [ex for ex in examples if ex['risk_level'] == risk_level]
        if matching_examples:
            example = matching_examples[0]
            prompt += f"{risk_level} Risk Example:\n"
            prompt += f"Patient information:\n{example['patient_text']}\n\n"
            prompt += "Reasoning:\n"
            prompt += "1. Key risk factors: " + example.get('risk_factors', 'None identified') + "\n"
            prompt += "2. Protective factors: " + example.get('protective_factors', 'None identified') + "\n"
            prompt += "3. Risk assessment: " + example['reasoning'] + "\n\n"
    
    # Add target patient with chain-of-thought structure
    prompt += f"New patient to assess:\n{patient_text}\n\n"
    prompt += """
    Reasoning (think step by step):
    1. Identify key risk factors for diabetes
    2. Identify any protective factors
    3. Assess the severity of each factor
    4. Consider interactions between factors
    5. Determine overall risk level

    Final Assessment: [Low/Medium/High] risk because...
    """
    return prompt

def create_examples(patient_df) -> List[Dict[str, Any]]:
    """
    Create example diabetes risk assessments for in-context learning.
    
    Args:
        patient_df: DataFrame containing patient data
        
    Returns:
        List of example assessments
    """
    # Create examples manually or select from dataset
    examples = []
    
    # Get one patient from each risk category if available
    for risk_level in ['Low', 'Medium', 'High']:
        category_patients = patient_df[patient_df['diabetes_risk_category'] == risk_level]
        if not category_patients.empty:
            # Take the first patient from each category
            patient = category_patients.iloc[0].to_dict()
            
            # Create reasoning based on patient data
            reasoning = create_reasoning_for_example(patient, risk_level)
            
            examples.append({
                'patient_text': prepare_patient_text(patient),
                'risk_level': risk_level,
                'reasoning': reasoning,
                'risk_factors': extract_risk_factors(patient),
                'protective_factors': extract_protective_factors(patient)
            })
    
    return examples

def create_reasoning_for_example(patient: Dict[str, Any], risk_level: str) -> str:
    """Create reasoning text for an example patient."""
    if risk_level == 'High':
        if patient['has_diabetes'] or patient['has_prediabetes']:
            return "they already have diabetes or prediabetes"
        elif patient.get('hba1c') and patient['hba1c'] >= 6.5:
            return "their HbA1c of {:.1f}% is at or above the diabetes threshold of 6.5%".format(patient['hba1c'])
        elif patient.get('glucose') and patient['glucose'] >= 200:
            return "their blood glucose of {:.0f} mg/dL is at or above the diabetes threshold of 200 mg/dL".format(patient['glucose'])
        else:
            return "they have multiple significant risk factors including {}".format(extract_risk_factors(patient))
    
    elif risk_level == 'Medium':
        if patient.get('hba1c') and 5.7 <= patient['hba1c'] < 6.5:
            return "their HbA1c of {:.1f}% is in the prediabetic range of 5.7-6.4%".format(patient['hba1c'])
        elif patient.get('glucose') and 140 <= patient['glucose'] < 200:
            return "their blood glucose of {:.0f} mg/dL is in the prediabetic range of 140-199 mg/dL".format(patient['glucose'])
        elif patient['has_obesity'] or (patient.get('bmi') and patient['bmi'] >= 30):
            return "they have obesity, which significantly increases diabetes risk"
        elif patient['has_hypertension']:
            return "they have hypertension, which is associated with increased diabetes risk"
        else:
            return "they have some risk factors including {}".format(extract_risk_factors(patient))
    
    else:  # Low
        protective_factors = extract_protective_factors(patient)
        if protective_factors:
            return "they have minimal risk factors and several protective factors, including " + protective_factors
        else:
            return "they have minimal risk factors for type 2 diabetes"

def extract_risk_factors(patient: Dict[str, Any]) -> str:
    """Extract diabetes risk factors from patient data."""
    factors = []
    
    if patient.get('age') and patient['age'] >= 45:
        factors.append("age over 45")
    
    if patient.get('bmi') and patient['bmi'] >= 25:
        if patient['bmi'] >= 30:
            factors.append("obesity (BMI {:.1f})".format(patient['bmi']))
        else:
            factors.append("overweight (BMI {:.1f})".format(patient['bmi']))
    
    if patient.get('has_hypertension') or (patient.get('systolic') and patient['systolic'] >= 140) or (patient.get('diastolic') and patient['diastolic'] >= 90):
        factors.append("hypertension")
    
    if patient.get('has_prediabetes'):
        factors.append("prediabetes")
    
    if patient.get('hba1c') and patient['hba1c'] >= 5.7:
        factors.append("elevated HbA1c ({:.1f}%)".format(patient['hba1c']))
    
    if patient.get('glucose') and patient['glucose'] >= 100:
        factors.append("elevated blood glucose ({:.0f} mg/dL)".format(patient['glucose']))
    
    if not factors:
        return "none identified"
    
    if len(factors) == 1:
        return factors[0]
    
    return ", ".join(factors[:-1]) + ", and " + factors[-1]

def extract_protective_factors(patient: Dict[str, Any]) -> str:
    """Extract protective factors against diabetes from patient data."""
    factors = []
    
    if patient.get('age') and patient['age'] < 40:
        factors.append("younger age")
    
    if patient.get('bmi') and patient['bmi'] < 25:
        factors.append("healthy BMI ({:.1f})".format(patient['bmi']))
    
    if patient.get('hba1c') and patient['hba1c'] < 5.7:
        factors.append("normal HbA1c ({:.1f}%)".format(patient['hba1c']))
    
    if patient.get('glucose') and patient['glucose'] < 100:
        factors.append("normal blood glucose ({:.0f} mg/dL)".format(patient['glucose']))
    
    if not factors:
        return ""
    
    if len(factors) == 1:
        return factors[0]
    
    return ", ".join(factors[:-1]) + ", and " + factors[-1]

# Import the prepare_patient_text function from data_processing
from src.data_processing import prepare_patient_text
