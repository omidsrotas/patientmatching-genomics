Simulacrum data release, version 2.1.0-2016-2019, 2023-04-21

This is a package of synthetic data based on cancer registration data from 
the National Cancer Registration Dataset (NCRD), the Systemic Anti-Cancer 
Treatment (SACT) dataset, the Radiotherapy Dataset (RTDS) and genomics testing 
data collected by the National Disease Registration Service (NDRS), NHS England. 
The synthetic data was developed by Health Data Insight CiC.

Any analysis performed on this synthetic data will not produce outputs that 
correspond exactly to analysis performed on the real data. The synthetic data 
does not contain real patients but just mimics the data structure and specific 
statistical properties of the real data. To minimise the potential for confusion 
with real data, organisational codes in this extract have been obfuscated.

For more information on the Simulacrum and guidance on use, please see
  https://simulacrum.healthdatainsight.org.uk/

If you would like access to detailed, quality assured statistical cancer data, 
please read the 'Using the Simulacrum' section of the above website. You can also 
find more information about the underlying datasets at 
  https://www.cancerdata.nhs.uk/

This package contains:
- the synthetic data in CSV format
- the lookup table excel file for data variables in CSV and Excel format 
- the data dictionary for the individual synthetic data tables
- README.txt: this textfile
- LICENSE.txt: textfile containing license for use of this release.

Folders:
data
------------------------------------------------------------------------------
- Synthetic cancer registration, genomics, SACT and RTDS data for 2016-2019 
  cancer diagnoses, as CSV files:
	sim_av_gene.csv
	sim_av_tumour.csv
	sim_av_patient.csv
	sim_rtds_combined.csv
	sim_rtds_episode.csv
	sim_rtds_prescription.csv
	sim_rtds_exposure.cav
	sim_sact_cycle.csv
	sim_sact_drug_detail.csv
	sim_sact_outcome.csv
	sim_sact_regimen.csv

Documents
------------------------------------------------------------------------------
- An excel workbook containing all the lookup tables:
	all_z_lookup_tables.xlsx
- An excel workbook containing a data dictionary for each table
	Simulacrum_2_data_dictionaryVer2.1.0_2016-2019.xlsx
