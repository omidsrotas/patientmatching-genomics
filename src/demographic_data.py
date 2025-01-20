import os
import pickle
import pandas as pd
import numpy as np
import json

data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'

patient_pd = pd.read_csv(data_dir + 'sim_av_patient.csv')
tumour_pd = pd.read_csv(data_dir + 'sim_av_tumour.csv')
regimen_pd = pd.read_csv(data_dir + 'sim_sact_regimen.csv')
cancer_types = pd.read_json(data_dir + 'cancer_types.json')




# extract the data based on cancer types and ICD10 codes
cancer_codes = cancer_types['ICD10 Code'].unique().tolist()
tumour_ls = []
for code in cancer_codes:
    cancer = cancer_types[cancer_types['ICD10 Code'] == code]['Cancer Type'].values[0]
    tumour_pd_temp = tumour_pd[tumour_pd['SITE_ICD10_O2_3CHAR'] == code]
    tumour_pd_temp['Cancer Type'] = cancer
    tumour_ls.append(tumour_pd_temp)

tumour_ls = pd.concat(tumour_ls)

# Demographics
tumour_demographic = tumour_ls[['GENDER', 'PATIENTID', 'Cancer Type','AGE']].drop_duplicates()
regimen_demographic = regimen_pd[['ENCORE_PATIENT_ID', 'HEIGHT_AT_START_OF_REGIMEN','WEIGHT_AT_START_OF_REGIMEN']]


# Merge the dataframes
merged_demographic = pd.merge(tumour_demographic, regimen_demographic, left_on='PATIENTID', right_on='ENCORE_PATIENT_ID', how='inner')
merged_demographic.drop_duplicates(subset=['PATIENTID'], inplace=True)
merged_demographic.dropna(subset=['HEIGHT_AT_START_OF_REGIMEN', 'WEIGHT_AT_START_OF_REGIMEN'], inplace=True)

# add ethnicity to the the merged_demographic
# read ethnicity data from the json file
with open(data_dir + 'ethnicity.json', 'r') as json_file:
    Ethnicity = json.load(json_file)
# apply the mapping from the Ethnicity dictionary to the patient_pd dataframe ETHNICITY column
# Create a reverse mapping dictionary
reverse_mapping = {code: ethnicity for ethnicity, codes in Ethnicity.items() for code in codes}

# Apply the reverse mapping to the ETHNICITY column
patient_pd['ETHNICITY'] = patient_pd['ETHNICITY'].map(reverse_mapping)

merged_demographic = merged_demographic.merge(patient_pd[['PATIENTID', 'ETHNICITY']], on='PATIENTID', how='left')

# save the data as a pickle file
patient_data = {}
patient_data['demographic'] = merged_demographic
# save the data patient data as a dictionary pickle file, first check if the file in directory exists


patient_data_path = data_dir + 'patient_data.pkl'

if not os.path.exists(patient_data_path):
    # Save the dictionary as a pickle file
    with open(patient_data_path, 'wb') as file:
        pickle.dump(patient_data, file)
