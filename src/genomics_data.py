import pandas as pd
import pickle

data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'

gene_pd = pd.read_csv(data_dir + 'sim_av_gene.csv')
# read the patient data as a pickle file (data dictionary)
patient_data_path = data_dir + 'patient_data.pkl'
with open(patient_data_path, 'rb') as file:
    patient_data = pickle.load(file)

patient_data['demographic']
gene_pd_matched = gene_pd[gene_pd['PATIENTID'].isin(patient_data['demographic']['PATIENTID'])]
gene_pd_filtered = gene_pd_matched[['TUMOURID','PATIENTID','GENE_DESC', 'EXP_GAT','DNASEQ_GAT']].dropna()

# select all the columns exept the DNA_SEQ_GAT
gene_pd_filtered_GENE = gene_pd_filtered.drop(columns=['DNASEQ_GAT'])
# find unique genes
unique_genes = gene_pd_filtered_GENE['GENE_DESC'].unique().tolist()
# for each patient ID make additional columns for each gene and fill that gene value with EXP_GAT
for gene in unique_genes:
    gene_pd_filtered_GENE[gene] = gene_pd_filtered_GENE['GENE_DESC'].apply(lambda x: 1 if x == gene else -1)
    # where gene_pd_filtered_GENE not equal to -1 multiply the EXP_GAT with the gene value
    gene_pd_filtered_GENE[gene] = gene_pd_filtered_GENE[gene] * gene_pd_filtered_GENE['EXP_GAT']
# for each column in the gene_pd_filtered_GENE dataframe, if the value is '' replace it with -1 in for loop over the columns
for col in gene_pd_filtered_GENE.columns:
    gene_pd_filtered_GENE[col] = gene_pd_filtered_GENE[col].apply(lambda x: -1 if x == '' else x)
# add exp to each gene name
gene_pd_filtered_GENE.columns = [col + '_EXP' if col in unique_genes else col for col in gene_pd_filtered_GENE.columns]
# drop GENE_DESC	EXP_GAT	columns
gene_pd_filtered_GENE.drop(columns=['GENE_DESC', 'EXP_GAT'], inplace=True)

# do the same as above expect for the DNA_SEQ_GAT
gene_pd_filtered_DNA = gene_pd_filtered.drop(columns=['EXP_GAT'])
unique_genes = gene_pd_filtered_DNA['GENE_DESC'].unique().tolist()
for gene in unique_genes:
    gene_pd_filtered_DNA[gene] = gene_pd_filtered_DNA['GENE_DESC'].apply(lambda x: 1 if x == gene else -1)
    gene_pd_filtered_DNA[gene] = gene_pd_filtered_DNA[gene] * gene_pd_filtered_DNA['DNASEQ_GAT']
for col in gene_pd_filtered_DNA.columns:
    gene_pd_filtered_DNA[col] = gene_pd_filtered_DNA[col].apply(lambda x: -1 if x == '' else x)
gene_pd_filtered_DNA.columns = [col + '_DNA' if col in unique_genes else col for col in gene_pd_filtered_DNA.columns]
gene_pd_filtered_DNA.drop(columns=['GENE_DESC', 'DNASEQ_GAT'], inplace=True)

# sort all gene_pd_filtered_GENE and gene_pd_filtered_DNA by TUMOURID and PATIENTID
gene_pd_filtered_GENE.sort_values(by=['TUMOURID', 'PATIENTID'], inplace=True)
gene_pd_filtered_DNA.sort_values(by=['TUMOURID', 'PATIENTID'], inplace=True)
# drop TUMOURID and PATIENTID columns
gene_pd_filtered_GENE.drop(columns=['TUMOURID', 'PATIENTID'], inplace=True)
# concatenate gene_pd_filtered_GENE and gene_pd_filtered_DNA
gene_pd_merged = pd.concat([gene_pd_filtered_GENE, gene_pd_filtered_DNA], axis=1)


# encoding the gene data as a dictionary
columns = gene_pd_merged.columns.tolist()
columns.remove('TUMOURID')
columns.remove('PATIENTID')
unique_values = []
for cl in columns:
    uni_cl_val = gene_pd_merged[cl].unique()
    for val in uni_cl_val:
        if val not in unique_values:
            unique_values.append(val)
# create a dictionary for unique_values
unique_values_dict = {}
for i, val in enumerate(unique_values):
    unique_values_dict[val] = i
unique_values_dict[-1]= -1
# for every column, map the unique values to integers
for cl in columns:
    gene_pd_merged[cl] = gene_pd_merged[cl].map(unique_values_dict)
# aggregating each patient data into a single row
patient_id = gene_pd_merged['PATIENTID'].unique()
gene_aggr_ls = []
for ptn_id in patient_id:
    gene_ptn = gene_pd_merged[gene_pd_merged['PATIENTID'] == ptn_id]
    # in each column set the value to the maximum of the gene expression
    gene_ptn = gene_ptn.agg('max')
    gene_ptn = pd.DataFrame(gene_ptn).T
    gene_aggr_ls.append(gene_ptn)
gene_pd_merged = pd.concat(gene_aggr_ls)

# save the patient data as a dictionary pickle file
patient_data['genomics'] = gene_pd_merged
with open(patient_data_path, 'wb') as file:
    pickle.dump(patient_data, file)