import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client
from azureml.core import Workspace, Dataset, Datastore


# read the drug data
data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'
drug_pd = pd.read_csv(data_dir + 'sim_sact_drug_detail.csv')

# small molecule drugs SMILES and fingerprints information extraction
drugs_name = drug_pd['DRUG_GROUP'].dropna().unique().tolist()
# extract the chemical structure of the drugs and drug fingerprints in pandas dataframe
small_molecules = []
drug_pd_small = pd.DataFrame(columns=['DRUG_GROUP', 'SMILES', 'FINGERPRINT'])
drug_pd_small['DRUG_GROUP'] = drugs_name
drug_pd_small['SMILES'] = np.nan
drug_pd_small['FINGERPRINT'] = np.nan
no_compound_list = []
for drug in drugs_name:
    drug_name_split = drug.split(' ')
    for drug_name in drug_name_split:
        if drug_name[0] == '(':
            drug_name = drug_name[1:-1]
    compound = pcp.get_compounds(drug_name, 'name')
    if len(compound) == 0:
        #print('No compound found for drug: ', drug)
        no_compound_list.append(drug)
        continue
    for c in compound:
        drug_pd_small.loc[drug_pd_small['DRUG_GROUP'] == drug_name, 'SMILES'] = c.isomeric_smiles
        drug_pd_small.loc[drug_pd_small['DRUG_GROUP'] == drug_name, 'FINGERPRINT'] = c.cactvs_fingerprint
        small_molecules.append(drug_pd_small)
small_molecules = pd.concat(small_molecules).dropna(subset = ['SMILES', 'FINGERPRINT'])


# extract the the large molecule sequence and component type in pandas dataframe
large_molecules = []
drug_pd_large = pd.DataFrame(columns=['DRUG_GROUP', 'SEQUENCE', 'COMPONENT_TYPE'])
drug_pd_large['DRUG_GROUP'] = drugs_name
no_molecule_list = []
molecule = new_client.molecule
for mol in no_compound_list:
    mol_name_split = mol.split(' ')
    for mol_name in mol_name_split:
        if mol_name[0] == '(':
            mol_name = mol_name[1:-1]
        results = molecule.filter(pref_name=mol_name)
        if len(results) == 0:
            no_molecule_list.append(mol_name)
            continue
        elif (results[0]['biotherapeutic'] is None) or (len(results[0]['biotherapeutic']['biocomponents']) == 0):
            no_molecule_list.append(mol_name)
            continue
        else:
            sequence = results[0]['biotherapeutic']['biocomponents'][0]['sequence']
            component_type = results[0]['biotherapeutic']['biocomponents'][0]['component_type']
            drug_pd_small.loc[drug_pd_small['DRUG_GROUP'] == mol_name, 'SEQUENCE'] = sequence
            drug_pd_small.loc[drug_pd_small['DRUG_GROUP'] == mol_name, 'COMPONENT_TYPE'] = component_type
            large_molecules.append(drug_pd_small)

large_molecules = pd.concat(large_molecules).dropna(subset = ['SEQUENCE', 'COMPONENT_TYPE'])

drug_data_path = data_dir + 'drug_data.pkl'
drug_data = {}
drug_data['small_molecules'] = small_molecules
drug_data['large_molecules'] = large_molecules
if not os.path.exists(drug_data_path):
    # Save the dictionary as a pickle file
    with open(drug_data_path, 'wb') as file:
        pickle.dump(drug_data, file)