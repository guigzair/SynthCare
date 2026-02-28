import pandas as pd


admissions = pd.read_csv('Data/ADMISSIONS.csv')
patients = pd.read_csv('Data/PATIENTS.csv')
diagnosis = pd.read_csv('Data/D_ICD_DIAGNOSES.csv')
diagnosis_procedure = pd.read_csv('Data/D_ICD_PROCEDURES.csv')

def create_clinical_story(patients_row, admissions_row, diagnosis_row, procedure_row):
    return f"""
    ### PATIENT PROFILE
    - ID: {patients_row['subject_id']}
    - Demographics: {patients_row['gender']}, ethnicity: {admissions_row['ethnicity']}
    - language: {admissions_row['language']}

    ### AMISSION PROFILE
    Admission Date: {admissions_row['admittime']}
    Discharge Date: {admissions_row['dischtime']}
    Location: {admissions_row['admission_location']}
    insurance: {admissions_row['insurance']}
    Death Indicator: {'Yes' if not pd.isna(admissions_row['deathtime']) else 'No'}

    ### CLINICAL EVENTS
    The patient was admitted with a primary diagnosis of {diagnosis_row['short_title']} or {diagnosis_row['long_title']}. 
    Secondary conditions identified: {diagnosis_row['icd9_code']}.
    Procedures performed: {procedure_row['short_title']} or {procedure_row['long_title']}.
    """

def instruction():
    return """
    You are a medical assistant. Your task is to read the clinical story of a patient and provide a concise summary of their medical history, including key diagnoses, procedures, and any relevant demographic information. Focus on the most critical aspects of the patient's health profile to assist healthcare professionals in understanding their case quickly.
    """

def context():
    return """
    The clinical story includes details about the patient's demographics, admission information, primary and secondary diagnoses, and procedures performed during their hospital stay. Use this information to generate a clear and informative summary that highlights the patient's medical history and current health status.
    """
    

# Apply to your merged dataframe
print(create_clinical_story(patients.iloc[0], admissions.iloc[0], diagnosis.iloc[0], diagnosis_procedure.iloc[0]))

# dataset generation
def generate_dataset(patients, admissions, diagnosis, procedure):
    dataset = []
    for i in range(len(patients)):
        story = create_clinical_story(patients.iloc[i], admissions.iloc[i], diagnosis.iloc[i], procedure.iloc[i])
        JSONL = {
            "instruction": instruction(),
            "context": context(),
            "output": story
        }
        dataset.append(JSONL)
    return dataset

data = generate_dataset(patients, admissions, diagnosis, diagnosis_procedure)

#save data in jsonl format
import json
with open('clinical_stories.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')