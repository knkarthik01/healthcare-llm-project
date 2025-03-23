"""
Module for processing Synthea FHIR data for diabetes risk prediction.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

def load_patient_data(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load patient data from Synthea FHIR JSON files.
    
    Args:
        data_dir: Directory containing the Synthea FHIR data
        
    Returns:
        Dictionary mapping patient IDs to patient records
    """
    patients = {}
    
    # Find all patient JSON files in the FHIR directory
    patient_dir = os.path.join(data_dir, 'fhir')
    patient_files = [f for f in os.listdir(patient_dir) if f.endswith('.json')]
    
    # First pass: Identify patients
    for file in patient_files:
        with open(os.path.join(patient_dir, file), 'r') as f:
            data = json.load(f)
            
            # Check if this is a bundle with multiple resources
            if data.get('resourceType') == 'Bundle':
                entries = data.get('entry', [])
                for entry in entries:
                    resource = entry.get('resource', {})
                    # Process patient resources
                    if resource.get('resourceType') == 'Patient':
                        patient_id = resource.get('id')
                        if patient_id:
                            patients[patient_id] = {
                                'id': patient_id,
                                'demographics': extract_demographics(resource),
                                'conditions': [],
                                'observations': [],
                                'medications': []
                            }
            
            # Or if it's a single patient resource
            elif data.get('resourceType') == 'Patient':
                patient_id = data.get('id')
                if patient_id:
                    patients[patient_id] = {
                        'id': patient_id,
                        'demographics': extract_demographics(data),
                        'conditions': [],
                        'observations': [],
                        'medications': []
                    }
    
    # Second pass: collect all medical data for each patient
    for file in patient_files:
        with open(os.path.join(patient_dir, file), 'r') as f:
            data = json.load(f)
            
            # Handle bundle resources
            if data.get('resourceType') == 'Bundle':
                entries = data.get('entry', [])
                for entry in entries:
                    resource = entry.get('resource', {})
                    process_resource(resource, patients)
            else:
                # Handle single resources
                process_resource(data, patients)
    
    return patients

def process_resource(resource: Dict[str, Any], patients: Dict[str, Dict[str, Any]]) -> None:
    """Process a FHIR resource and add it to the appropriate patient record."""
    resource_type = resource.get('resourceType')
    
    # Skip patient resources (already processed)
    if resource_type == 'Patient':
        return
        
    # Get patient reference, if it exists
    patient_id = None
    if 'subject' in resource and 'reference' in resource['subject']:
        ref = resource['subject']['reference']
        if ref.startswith('Patient/'):
            patient_id = ref.replace('Patient/', '')
    
    if not patient_id or patient_id not in patients:
        return
        
    # Process based on resource type
    if resource_type == 'Condition':
        condition = extract_condition(resource)
        if condition:
            patients[patient_id]['conditions'].append(condition)
    
    elif resource_type == 'Observation':
        observation = extract_observation(resource)
        if observation:
            patients[patient_id]['observations'].append(observation)
    
    elif resource_type == 'MedicationRequest' or resource_type == 'MedicationStatement':
        medication = extract_medication(resource)
        if medication:
            patients[patient_id]['medications'].append(medication)

def extract_demographics(patient_resource: Dict[str, Any]) -> Dict[str, Any]:
    """Extract demographic information from a patient resource."""
    # Basic demographics
    gender = patient_resource.get('gender', 'unknown')
    
    # Calculate age from birthDate
    birth_date_str = patient_resource.get('birthDate')
    age = None
    if birth_date_str:
        try:
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except (ValueError, TypeError):
            pass
    
    # Get race and ethnicity if available
    race = None
    ethnicity = None
    extensions = patient_resource.get('extension', [])
    for ext in extensions:
        if ext.get('url') == 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-race':
            for race_ext in ext.get('extension', []):
                if race_ext.get('url') == 'text':
                    race = race_ext.get('valueString')
        
        if ext.get('url') == 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity':
            for eth_ext in ext.get('extension', []):
                if eth_ext.get('url') == 'text':
                    ethnicity = eth_ext.get('valueString')
    
    return {
        'gender': gender,
        'age': age,
        'race': race,
        'ethnicity': ethnicity
    }

def extract_condition(condition_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract condition information from a condition resource."""
    code_info = condition_resource.get('code', {})
    coding = code_info.get('coding', [{}])[0]
    
    # Get basic condition info
    condition_name = code_info.get('text', coding.get('display', 'Unknown Condition'))
    condition_code = coding.get('code')
    condition_system = coding.get('system')
    
    # Get clinical status
    clinical_status = 'unknown'
    if 'clinicalStatus' in condition_resource:
        status_coding = condition_resource['clinicalStatus'].get('coding', [{}])[0]
        clinical_status = status_coding.get('code', 'unknown')
    
    # Get onset date if available
    onset_date = None
    if 'onsetDateTime' in condition_resource:
        onset_date = condition_resource['onsetDateTime']
    
    return {
        'name': condition_name,
        'code': condition_code,
        'system': condition_system,
        'status': clinical_status,
        'onset_date': onset_date
    }

def extract_observation(observation_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract observation information from an observation resource."""
    code_info = observation_resource.get('code', {})
    coding = code_info.get('coding', [{}])[0]
    
    # Get basic observation info
    observation_name = code_info.get('text', coding.get('display', 'Unknown Observation'))
    observation_code = coding.get('code')
    observation_system = coding.get('system')
    
    # Get value
    value = None
    value_type = None
    
    if 'valueQuantity' in observation_resource:
        value = observation_resource['valueQuantity'].get('value')
        value_type = 'quantity'
        unit = observation_resource['valueQuantity'].get('unit', '')
    elif 'valueCodeableConcept' in observation_resource:
        value_coding = observation_resource['valueCodeableConcept'].get('coding', [{}])[0]
        value = value_coding.get('display', value_coding.get('code'))
        value_type = 'concept'
        unit = ''
    elif 'valueString' in observation_resource:
        value = observation_resource['valueString']
        value_type = 'string'
        unit = ''
    else:
        # If no direct value, check component values (e.g., BP has systolic/diastolic)
        components = observation_resource.get('component', [])
        if components:
            value = {}
            value_type = 'component'
            unit = ''
            for component in components:
                comp_code = component.get('code', {}).get('coding', [{}])[0].get('code')
                comp_value = component.get('valueQuantity', {}).get('value')
                if comp_code and comp_value is not None:
                    value[comp_code] = comp_value
    
    # Get the date of the observation
    effective_date = None
    if 'effectiveDateTime' in observation_resource:
        effective_date = observation_resource['effectiveDateTime']
    
    if value is None:
        return None
        
    return {
        'name': observation_name,
        'code': observation_code,
        'system': observation_system,
        'value': value,
        'value_type': value_type,
        'unit': unit,
        'date': effective_date
    }

def extract_medication(medication_resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract medication information from a medication resource."""
    # The medication details could be directly in the resource or in a referenced resource
    med_code_info = None
    
    if 'medicationCodeableConcept' in medication_resource:
        med_code_info = medication_resource['medicationCodeableConcept']
    elif 'code' in medication_resource:
        med_code_info = medication_resource['code']
    
    if not med_code_info:
        return None
        
    coding = med_code_info.get('coding', [{}])[0]
    
    # Get basic medication info
    medication_name = med_code_info.get('text', coding.get('display', 'Unknown Medication'))
    medication_code = coding.get('code')
    medication_system = coding.get('system')
    
    # Get status if available
    status = medication_resource.get('status', 'unknown')
    
    # Get the date
    authored_date = None
    if 'authoredOn' in medication_resource:
        authored_date = medication_resource['authoredOn']
    
    return {
        'name': medication_name,
        'code': medication_code,
        'system': medication_system,
        'status': status,
        'date': authored_date
    }

def create_patient_dataframe(patients: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert patient dictionary to DataFrame with diabetes risk factors.
    
    Args:
        patients: Dictionary of patient records
        
    Returns:
        DataFrame with patient data and diabetes risk factors
    """
    patient_rows = []
    
    for patient_id, patient in patients.items():
        # Start with demographics
        row = {
            'id': patient_id,
            'gender': patient['demographics'].get('gender'),
            'age': patient['demographics'].get('age'),
            'race': patient['demographics'].get('race'),
            'ethnicity': patient['demographics'].get('ethnicity')
        }
        
        # Extract key observations
        row['weight'] = extract_latest_observation_value(patient['observations'], 'body weight')
        row['height'] = extract_latest_observation_value(patient['observations'], 'body height')
        row['bmi'] = extract_latest_observation_value(patient['observations'], 'BMI')
        row['glucose'] = extract_latest_observation_value(patient['observations'], 'glucose')
        row['hba1c'] = extract_latest_observation_value(patient['observations'], 'hemoglobin A1c')
        row['systolic'] = extract_latest_observation_value(patient['observations'], 'systolic blood pressure')
        row['diastolic'] = extract_latest_observation_value(patient['observations'], 'diastolic blood pressure')
        
        # Calculate BMI if not directly available
        if row['bmi'] is None and row['weight'] is not None and row['height'] is not None:
            # Convert height from cm to m if needed
            height_m = row['height'] / 100 if row['height'] > 3 else row['height']
            row['bmi'] = row['weight'] / (height_m ** 2)
        
        # Check for diabetes-related conditions
        row['has_diabetes'] = check_for_condition(patient['conditions'], 'diabetes')
        row['has_prediabetes'] = check_for_condition(patient['conditions'], 'prediabetes')
        row['has_gestational_diabetes'] = check_for_condition(patient['conditions'], 'gestational diabetes')
        row['has_hypertension'] = check_for_condition(patient['conditions'], 'hypertension')
        row['has_obesity'] = check_for_condition(patient['conditions'], 'obesity')
        
        # Extract medication names
        row['medications'] = [med['name'] for med in patient['medications']]
        
        # Calculate diabetes risk category
        # Simple heuristic:
        # - High: Has diabetes/prediabetes/HbA1c >= 6.5/glucose >= 200/multiple risk factors
        # - Medium: HbA1c 5.7-6.4/glucose 140-199/obesity/hypertension
        # - Low: No significant risk factors
        
        if row['has_diabetes'] or row['has_prediabetes'] or row['has_gestational_diabetes']:
            row['diabetes_risk_category'] = 'High'
        elif (row['hba1c'] is not None and row['hba1c'] >= 6.5) or (row['glucose'] is not None and row['glucose'] >= 200):
            row['diabetes_risk_category'] = 'High'
        elif (row['hba1c'] is not None and row['hba1c'] >= 5.7) or (row['glucose'] is not None and row['glucose'] >= 140):
            row['diabetes_risk_category'] = 'Medium'
        elif row['has_obesity'] or row['has_hypertension'] or (row['bmi'] is not None and row['bmi'] >= 30):
            row['diabetes_risk_category'] = 'Medium'
        else:
            row['diabetes_risk_category'] = 'Low'
        
        patient_rows.append(row)
    
    df = pd.DataFrame(patient_rows)
    return df

def extract_latest_observation_value(observations: List[Dict[str, Any]], name_substring: str) -> Optional[float]:
    """Extract the most recent value for a specific observation type."""
    relevant_obs = [
        obs for obs in observations 
        if name_substring.lower() in obs['name'].lower() and
        obs['value_type'] in ['quantity', 'component'] and
        obs['value'] is not None
    ]
    
    if not relevant_obs:
        return None
    
    # Sort by date (most recent first)
    relevant_obs.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    # Get the most recent observation value
    most_recent = relevant_obs[0]
    
    # Handle different value types
    if most_recent['value_type'] == 'quantity':
        return float(most_recent['value'])
    elif most_recent['value_type'] == 'component':
        # For component types, return the first numeric value (e.g., systolic for BP)
        for value in most_recent['value'].values():
            if value is not None:
                return float(value)
    
    return None

def check_for_condition(conditions: List[Dict[str, Any]], name_substring: str) -> bool:
    """Check if patient has a condition containing the given substring."""
    return any(name_substring.lower() in condition['name'].lower() for condition in conditions)

def prepare_patient_text(patient: Dict[str, Any]) -> str:
    """
    Convert patient record to text description for LLM input.
    
    Args:
        patient: Dictionary containing patient data
        
    Returns:
        Text description of patient for use in LLM prompts
    """
    lines = []
    
    # ID and demographics
    lines.append(f"Patient ID: {patient['id']}")
    
    # Format demographics
    if patient['age'] is not None:
        lines.append(f"Age: {patient['age']} years")
    lines.append(f"Gender: {patient['gender']}")
    if patient['race'] is not None:
        lines.append(f"Race: {patient['race']}")
    
    # Format key vitals and measurements
    if patient['weight'] is not None:
        lines.append(f"Weight: {patient['weight']:.1f} kg")
    if patient['height'] is not None:
        # Convert height to cm if in meters
        height = patient['height']
        unit = 'cm'
        if height < 3:  # Probably in meters
            height = height * 100
        lines.append(f"Height: {height:.1f} {unit}")
    if patient['bmi'] is not None:
        lines.append(f"BMI: {patient['bmi']:.1f} kg/m²")
    
    # Blood pressure
    if patient['systolic'] is not None and patient['diastolic'] is not None:
        lines.append(f"Blood Pressure: {patient['systolic']:.0f}/{patient['diastolic']:.0f} mmHg")
    
    # Lab values relevant to diabetes
    if patient['glucose'] is not None:
        lines.append(f"Blood Glucose: {patient['glucose']:.1f} mg/dL")
    if patient['hba1c'] is not None:
        lines.append(f"HbA1c: {patient['hba1c']:.1f}%")
    
    # Medical conditions
    conditions = []
    if patient['has_diabetes']:
        conditions.append("Diabetes")
    if patient['has_prediabetes']:
        conditions.append("Prediabetes")
    if patient['has_gestational_diabetes']:
        conditions.append("Gestational Diabetes History")
    if patient['has_hypertension']:
        conditions.append("Hypertension")
    if patient['has_obesity']:
        conditions.append("Obesity")
    
    if conditions:
        lines.append(f"Medical Conditions: {', '.join(conditions)}")
    else:
        lines.append("Medical Conditions: None significant for diabetes risk")
    
    # Medications
    med_list = patient.get('medications', [])
    if med_list and len(med_list) > 0:
        # Limit to 5 medications to keep prompt size reasonable
        if len(med_list) > 5:
            med_text = ', '.join(med_list[:5]) + f", and {len(med_list)-5} others"
        else:
            med_text = ', '.join(med_list)
        lines.append(f"Medications: {med_text}")
    else:
        lines.append("Medications: None")
    
    return '\n'.join(lines)
