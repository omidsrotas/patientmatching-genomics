import numpy as np
import pandas as pd
import pickle
import torch
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# stratified sampling based on the patients ethinicity
import random
from torch.utils.data import Sampler
from collections import defaultdict
import yaml
import logging
import random
import re



# combine the DemographicDataset and GenomicsDataset into one dataset class
# combine the DemographicDataset and GenomicsDataset into one dataset class
class PatientDataset(Dataset):
    def __init__(self, genomics_data, demographic_data,labels_df,
                 small_molecule_embeddings, large_molecule_embeddings,
                  cancer_types, patient_ids, ethnicity):
        self.genomics_data = genomics_data
        self.demographic_data = demographic_data
        self.cancer_types = cancer_types
        self.patient_ids = patient_ids
        self.labels_df = labels_df
        self.small_molecule_embeddings = small_molecule_embeddings
        self.large_molecule_embeddings = large_molecule_embeddings
        self.ethnicity = {key[0]: i  for i, key in enumerate(ethnicity.items())}
    def __len__(self):
        return len(self.patient_ids)
    
    def process_demographics(self):
        # divide age by 50 --> normalize the age by 50
        # divide weight by 100 --> normalize the age by 100
        self.demographic_data['AGE'] = self.demographic_data['AGE'] / 50
        self.demographic_data['WEIGHT_AT_START_OF_REGIMEN'] = self.demographic_data['WEIGHT_AT_START_OF_REGIMEN'] / 100

    def extract_drug_embedding_and_label(self, idx):
        label_patients = self.labels_df[self.labels_df['ENCORE_PATIENT_ID'].isin([idx])]
        label_patients.drop_duplicates(subset=['ENCORE_PATIENT_ID'], inplace=True)
        y_label = label_patients['Label'].values
        # Explicit sizes for small and large molecule embeddings
        SMALL_MOLECULE_SIZE = 768
        LARGE_MOLECULE_SIZE = 1024
        drug_embed_large = np.zeros(LARGE_MOLECULE_SIZE)
        drug_embed_small = np.zeros(SMALL_MOLECULE_SIZE)
        if len(label_patients['BENCHMARK_GROUP'].tolist()) > 0:
            treatment = label_patients['BENCHMARK_GROUP'].tolist()[0]
            # Extract only the string parts from treatment
            drugs = re.findall(r'\b\w+\b', treatment)
            for drug in drugs:
                if drug not in self.small_molecule_embeddings.keys():
                    " extract the drug embedding from the large molecule embeddings"
                    if drug  in self.large_molecule_embeddings.keys():
                        drug_embed_large += self.large_molecule_embeddings[drug]
                else:
                    " extract the drug embedding from the small molecule embeddings"
                    drug_embed_small += self.small_molecule_embeddings[drug]       
        else:
            y_label = np.array([])
        return drug_embed_small,drug_embed_large, y_label

    def __getitem__(self, idx):
        self.process_demographics()
        #print("idx: ", idx)
        patient_genome = self.genomics_data[self.genomics_data['PATIENTID'].isin([idx])]
        patient_genome.drop(columns=['PATIENTID',  'TUMOURID', 'ETHNICITY'], inplace=True)
        patient_genome = patient_genome.values
        
        patient_demographics = self.demographic_data[self.demographic_data['PATIENTID'].isin([idx])]
        patient_demographics.drop(columns=['PATIENTID',  'ENCORE_PATIENT_ID'], inplace=True)
        patient_demographics['Cancer Type'] = patient_demographics['Cancer Type'].apply(lambda x: self.cancer_types[x])
        # convert ethnicity to integer using the ethnicity dictionary
        #patient_demographics['ETHNICITY'] = patient_demographics['ETHNICITY'].map(self.ethnicity)
        patient_demographics.drop(columns=['ETHNICITY'], inplace=True)
        #patient_demographics['Ethnicity'] = patient_demographics['Ethnicity'].apply(lambda x: self.cancer_types[x])
        patient_demographics = patient_demographics.values
        drug_embed_small,drug_embed_large, y_label = self.extract_drug_embedding_and_label(idx)
        # if drug_embedding is 0, then do not include the patient in the dataset
        #if not y_label.any == 0:
        #    y_label = -1
        #    drug_embed_small = -1*np.ones_like(drug_embed_small)
        #    drug_embed_large = -1*np.ones_like(drug_embed_large)
        patient_data = {'patient_id': idx,
                        'genome': patient_genome,
                        'demographics': patient_demographics,
                        'drug_embed_small': drug_embed_small,
                        'drug_embed_large': drug_embed_large,
                        'label': y_label}
        return patient_data

# Define the BalancedBatchSampler class
class EthnicityBalancedSampler(Sampler):
    def __init__(self, genomics_data, batch_size):
        self.genomics_data = genomics_data
        self.batch_size = batch_size

        # Group indices by ehtnicity
        self.groups = self.genomics_data.groupby('ETHNICITY')['PATIENTID'].apply(list).to_dict()
        # ensure batch size can accomodate all groups
        assert self.batch_size >= len(self.groups.keys()), (
            'Batch size must be greater than or equal to the number of groups'
        )
    def __iter__(self):
        indices = []
        # Shuffle each ethinicity group
        group_indices = {eth: random.sample(indices, len(indices)) 
                         for eth, indices in self.groups.items()}
        
        # create batches
        while any(group_indices.values()):
            batch = []
            for eth, idx_list in group_indices.items():
                if idx_list:
                    batch.append(idx_list.pop()) # Take one sample from each group
            # Add additional samples to fill the batch size
            while len(batch) < self.batch_size:
                availabel_ethnicities = [eth for eth, idx_list in group_indices.items() if idx_list]
                if not availabel_ethnicities:
                    break
                eth = random.choice(availabel_ethnicities)
                batch.append(group_indices[eth].pop())
            yield batch
    
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.ethnicity_groups.values())
        return total_samples // self.batch_size

########################################################################################
# if __name__ == '__main__':

#     data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'
#     # read the genomics data as the pickle file
#     with open(data_dir + 'patient_data.pkl', 'rb') as f:
#         patient_data = pickle.load(f)

#     genomics_data = patient_data['genomics']
#     demographic_data = patient_data['demographic']
#     # fillna in the ETHINICITY with the most frequent value
#     demographic_data['ETHNICITY'].fillna('White', inplace=True)
#     # Add ETHNICITY to the genomics data using the patient_id
#     genomics_data = genomics_data.merge(demographic_data[['PATIENTID', 'ETHNICITY']], on='PATIENTID', how='left')

#     # Group the data by ethnicity
#     ethnicity_groups = demographic_data.groupby('ETHNICITY').apply(lambda x: x.index.tolist()).to_dict()

#     cancer_types = {}
#     for i, cancer in enumerate(demographic_data['Cancer Type'].unique().tolist()):
#         cancer_types[cancer] = i


#     # read ethnicity data from the json file
#     with open(data_dir + 'ethnicity.json', 'r') as json_file:
#         Ethnicity = json.load(json_file)

#     # Define batch size
#     batch_size = 16

#     # Create DataLoader
#     sampler = EthnicityBalancedSampler(genomics_data, batch_size)
#     # create a dataloader
#     patient_ids = genomics_data['PATIENTID'].unique().tolist()
#     patient_dataset_train = PatientDataset(genomics_data, demographic_data, cancer_types, patient_ids, Ethnicity)
#     patient_dataloader_train = DataLoader(patient_dataset_train, batch_sampler= sampler)
