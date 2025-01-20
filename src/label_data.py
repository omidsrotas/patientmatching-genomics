import pandas as pd
import numpy as np
import copy
import pickle
import os

data_dir = '/home/azureuser/cloudfiles/code/Users/Omid.Bazgir/data/'

outcomes_pd = pd.read_csv(data_dir + 'sim_sact_outcome.csv')
regimen_pd = pd.read_csv(data_dir + 'sim_sact_regimen.csv')
# load the regimen data
drug_filtered = regimen_pd.BENCHMARK_GROUP.value_counts()[:300].index.to_list()
regimen_pd_filtered = regimen_pd[regimen_pd['BENCHMARK_GROUP'].isin(drug_filtered)]

# strong labels
# save the outcome_label_pd as a pickle file
label_pickle = {}

# create a dictionary of labels based on the each regimen modification type
label_dict = {}
# dose reduction is an indication of toxicity and safety issues
dose_reduction = {'Y': 0, 'N': None} 
label_dict['REGIMEN_MOD_DOSE_REDUCTION'] = dose_reduction
# early stop is an indication of success of the treatment
early_stop = {'Y': 1, 'N': None} 
label_dict['REGIMEN_MOD_STOPPED_EARLY'] = early_stop

# time delay is an indication of safety and efficacy
time_delay = {'Y': 0, 'N': None}
label_dict['REGIMEN_MOD_TIME_DELAY'] = time_delay
# regimen outcome summary is an indication of the failure of the treatment according to the data specifications
outcome_summary_dict = {'0': None,'1':0, '2':0, '3':0, '4':0, '5': None}
label_dict['REGIMEN_OUTCOME_SUMMARY'] = outcome_summary_dict

# make strong labels for the outcomes
strong_outcome_label_pd = copy.copy(outcomes_pd)
for key in label_dict.keys():
    strong_outcome_label_pd[key] = strong_outcome_label_pd[key].map(label_dict[key])

label_pickle['strong_label'] = strong_outcome_label_pd

print("STRONG LABELS post processing")
strong_outcome_label_pd['Label'] = strong_outcome_label_pd['REGIMEN_OUTCOME_SUMMARY']
print(strong_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_DOSE_REDUCTION is 0, then the label is 0
# find where the label is NaN and REGIMEN_MOD_DOSE_REDUCTION is 0
index = strong_outcome_label_pd['Label'].isna() & (strong_outcome_label_pd['REGIMEN_MOD_DOSE_REDUCTION'] == 0)
strong_outcome_label_pd.loc[index, 'Label'] = 0
print(strong_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_TIME_DELAY is 0, then the label is 0
# find where the label is NaN and REGIMEN_MOD_TIME_DELAY is 0
index = strong_outcome_label_pd['Label'].isna() & (strong_outcome_label_pd['REGIMEN_MOD_TIME_DELAY'] == 0)
strong_outcome_label_pd.loc[index, 'Label'] = 0
print(strong_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_STOPPED_EARLY is 1, then the label is 1
# find where the label is NaN and REGIMEN_MOD_STOPPED_EARLY is 1
index = strong_outcome_label_pd['Label'].isna() & (strong_outcome_label_pd['REGIMEN_MOD_STOPPED_EARLY'] == 1)
strong_outcome_label_pd.loc[index, 'Label'] = 1
print(strong_outcome_label_pd['Label'].value_counts())
label_pickle['strong_label'] = strong_outcome_label_pd
# add the treatment information to the strong label data
# Extract 'strong_label' from label_data
strong_label = label_pickle['strong_label']
# Merge the DataFrames on 'merged_regimen_ID'
merged_data_strong = pd.merge(strong_label, regimen_pd_filtered, on='MERGED_REGIMEN_ID')
# Display the merged DataFrame
label_pickle['strong_label'] = merged_data_strong


# make weak labels for the outcomes
# dose reduction is an indication of toxicity and safety issues
dose_reduction = {'Y': 0, 'N': 1} 
label_dict['REGIMEN_MOD_DOSE_REDUCTION'] = dose_reduction
# early stop is an indication of success of the treatment
early_stop = {'Y': 1, 'N': 0} 
label_dict['REGIMEN_MOD_STOPPED_EARLY'] = early_stop

# time delay is an indication of safety and efficacy
time_delay = {'Y': 0, 'N': 1}
label_dict['REGIMEN_MOD_TIME_DELAY'] = time_delay
# regimen outcome summary is an indication of the failure of the treatment according to the data specifications
outcome_summary_dict = {'0': 1,'1':0, '2':0, '3':0, '4':0, '5': 1}
label_dict['REGIMEN_OUTCOME_SUMMARY'] = outcome_summary_dict

# make strong labels for the outcomes
weak_outcome_label_pd = copy.copy(outcomes_pd)
for key in label_dict.keys():
    weak_outcome_label_pd[key] = weak_outcome_label_pd[key].map(label_dict[key])

label_pickle['weak_label'] = weak_outcome_label_pd


print("WEAK LABELS post processing")
weak_outcome_label_pd['Label'] = weak_outcome_label_pd['REGIMEN_OUTCOME_SUMMARY']
print(weak_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_STOPPED_EARLY is 1, then the label is 1
# find where the label is NaN and REGIMEN_MOD_STOPPED_EARLY is 1
index = weak_outcome_label_pd['Label'].isna() & (weak_outcome_label_pd['REGIMEN_MOD_STOPPED_EARLY'] == 1)
weak_outcome_label_pd.loc[index, 'Label'] = 1
print(weak_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_DOSE_REDUCTION is 0, then the label is 0
# find where the label is NaN and REGIMEN_MOD_DOSE_REDUCTION is 0
index = weak_outcome_label_pd['Label'].isna() & (weak_outcome_label_pd['REGIMEN_MOD_DOSE_REDUCTION'] == 0)
weak_outcome_label_pd.loc[index, 'Label'] = 0
print(weak_outcome_label_pd['Label'].value_counts())
# if the label is NaN, and REGIMEN_MOD_TIME_DELAY is 0, then the label is 0
# find where the label is NaN and REGIMEN_MOD_TIME_DELAY is 0
index = weak_outcome_label_pd['Label'].isna() & (weak_outcome_label_pd['REGIMEN_MOD_TIME_DELAY'] == 0)
weak_outcome_label_pd.loc[index, 'Label'] = 0
print(weak_outcome_label_pd['Label'].value_counts())
weak_outcome_label_pd['Label'].fillna(0, inplace=True)
print(weak_outcome_label_pd['Label'].value_counts())
label_pickle['weak_label'] = weak_outcome_label_pd
# add the treatment information to the weak label data
# Extract 'weak_label' from label_data
weak_label = label_pickle['weak_label']
# Merge the DataFrames on 'merged_regimen_ID'
merged_data_weak = pd.merge(weak_label, regimen_pd_filtered, on='MERGED_REGIMEN_ID')
# Display the merged DataFrame
label_pickle['weak_label'] = merged_data_weak

# save the label_pickle as a pickle file
with open(data_dir + 'label_pickle.pkl', 'wb') as f:
    pickle.dump(label_pickle, f)