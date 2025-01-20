import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import yaml
import logging
import os
from sklearn.model_selection import train_test_split
from src.data_loader import *
from models.encoders import *
from models.twotower import *
import warnings
# set all the warnings off
warnings.filterwarnings("ignore")


data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'
# read the genomics data as the pickle file
with open(data_dir + 'patient_data.pkl', 'rb') as f:
    patient_data = pickle.load(f)


# read the label data and drug data as pickle files
with open(data_dir + 'label_pickle.pkl', 'rb') as f:
    label_data = pickle.load(f)
with open(data_dir + 'drug_data_embedding.pkl', 'rb') as f:
    drug_data = pickle.load(f)
small_molecule_embeddings = drug_data['small_molecule']
large_molecule_embeddings = drug_data['large_molecule']
labels_df = label_data['weak_label']

genomics_data = patient_data['genomics']
demographic_data = patient_data['demographic']
# fillna in the ETHINICITY with the most frequent value
demographic_data['ETHNICITY'].fillna('White', inplace=True)
# Add ETHNICITY to the genomics data using the patient_id
genomics_data = genomics_data.merge(demographic_data[['PATIENTID', 'ETHNICITY']], on='PATIENTID', how='left')

cancer_types = {}
for i, cancer in enumerate(demographic_data['Cancer Type'].unique().tolist()):
    cancer_types[cancer] = i

# read ethnicity data from the json file
with open(data_dir + 'ethnicity.json', 'r') as json_file:
    Ethnicity = json.load(json_file)

# read the config (yaml) file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config['genome_encoder']['input_dim'] = genomics_data.shape[1] - 3

cancer_types = {}
for i, cancer in enumerate(demographic_data['Cancer Type'].unique().tolist()):
    cancer_types[cancer] = i

# read ethnicity data from the json file
with open(data_dir + 'ethnicity.json', 'r') as json_file:
    Ethnicity = json.load(json_file)

# Set up logging
logging.basicConfig(filename='training_patient_matching.log', level=logging.INFO, format='%(asctime)s %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define batch size
batch_size = config['train_param']['batch_size']

# Create DataLoader

# create a dataloader
patient_ids = genomics_data['PATIENTID'].unique().tolist()
labels_df = labels_df[labels_df['ENCORE_PATIENT_ID'].isin(patient_ids)]
labels_df = labels_df.drop_duplicates(subset=['ENCORE_PATIENT_ID'], keep='first')
patient_ids = labels_df['ENCORE_PATIENT_ID'].unique().tolist()
# split the patient_ids into train and test
patient_train_ids, patient_test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
genomics_data_train = genomics_data[genomics_data['PATIENTID'].isin(patient_train_ids)]
demographic_data_train = demographic_data[demographic_data['PATIENTID'].isin(patient_train_ids)]
labels_df_train = labels_df[labels_df['ENCORE_PATIENT_ID'].isin(patient_train_ids)]


genomics_data_test = genomics_data[genomics_data['PATIENTID'].isin(patient_test_ids)]
demographic_data_test = demographic_data[demographic_data['PATIENTID'].isin(patient_test_ids)]
labels_df_test = labels_df[labels_df['ENCORE_PATIENT_ID'].isin(patient_test_ids)]
sampler_train = EthnicityBalancedSampler(genomics_data_train, batch_size)
patient_dataset_train = PatientDataset(genomics_data_train, demographic_data_train, 
                                       labels_df_train,small_molecule_embeddings, large_molecule_embeddings,
                                        cancer_types, patient_train_ids, Ethnicity)
patient_dataloader_train = DataLoader(patient_dataset_train, batch_sampler= sampler_train)

sampler_test = EthnicityBalancedSampler(genomics_data_test, batch_size)
patient_dataset_test = PatientDataset(genomics_data_test, demographic_data_test, 
                                      labels_df_test,small_molecule_embeddings, large_molecule_embeddings,
                                      cancer_types, patient_test_ids, Ethnicity)
patient_dataloader_test = DataLoader(patient_dataset_test, batch_sampler= sampler_test)

# Define the model
input_dim = config['genome_encoder']['input_dim']  # Example input dimension for genomics data
hidden_dim = config['genome_encoder']['hidden_dim']  # Example hidden layer dimension
latent_dim = config['genome_encoder']['latent_dim']   # Example latent space dimension
# Instantiate the VAE
genome_encoder = VAE(input_dim, hidden_dim, latent_dim, batch_size).to(device)
# read the weights of the model
weight_path = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/model_weights/genome_vae_model.pth'
genome_encoder.load_state_dict(torch.load(weight_path))

largeMolEncod = LargeMoleculeEncoder(config['largemolecule_encoder']['input_dim'],
                                    config['largemolecule_encoder']['hidden_dim'],
                                    config['largemolecule_encoder']['latent_dim']).to(device)
smallMolEncod = SmallMoleculeEncoder(config['smallmolecule_encoder']['input_dim'],
                                    config['smallmolecule_encoder']['hidden_dim'],
                                    config['smallmolecule_encoder']['latent_dim']).to(device)

# Instantiate the TwoTowerModel
patient_matching = TwoTowerModel(genome_encoder, largeMolEncod, smallMolEncod).to(device)
loss_fn = TwoTowerLoss(regularizer=0.1).to(device)

# define the optimizer
optimizer = torch.optim.Adam(patient_matching.parameters(), lr=config['train_param']['learning_rate'])
train_loss = []
val_loss = []
for epoch in range(config['train_param']['n_epochs']):
        patient_matching.train()
        train_epoch_loss = 0
        for batch in patient_dataloader_train:
                patient_treatment_similarity, match_score = patient_matching(batch, device)
                match_score_loss, contrastive_loss = loss_fn(patient_treatment_similarity, match_score, batch)
                optimizer.zero_grad()
                total_loss = match_score_loss + contrastive_loss
                train_epoch_loss += total_loss.item()
                total_loss.backward()
                optimizer.step()
        train_loss.append(train_epoch_loss)
        logging.info(f'Epoch {epoch}, Match Score Loss: {match_score_loss}, Contrastive Loss: {contrastive_loss}')
        if epoch % 10 == 0:
                print(f'Epoch {epoch}, Match Score Loss: {match_score_loss}, Contrastive Loss: {contrastive_loss}')
        # evaluate the model
        patient_matching.eval()
        val_epoch_loss = 0
        with torch.no_grad():
                for batch in patient_dataloader_test:
                        patient_treatment_similarity, match_score = patient_matching(batch, device)
                        match_score_loss, contrastive_loss = loss_fn(patient_treatment_similarity, match_score, batch)
                        total_loss = match_score_loss + contrastive_loss
                        val_epoch_loss += total_loss.item()
                val_loss.append(val_epoch_loss)
                logging.info(f'Epoch {epoch}, Test Match Score Loss: {match_score_loss}, Test Contrastive Loss: {contrastive_loss}')
                if epoch % 10 == 0:
                        print(f'Epoch {epoch}, Test Match Score Loss: {match_score_loss}, Test Contrastive Loss: {contrastive_loss}')
# plot and save the training loss and validation loss
plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig('twotower_loss.png')
# save the model weights
torch.save(patient_matching.state_dict(), '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/model_weights/patient_matching.pth')