import pandas as pd
from azureml.core import Dataset

# Assuming the previous code to list CSV files is already present
datastore_path = [(datastore, 'UI/2024-11-05_052811_UTC/simulacrum/')]
dataset = Dataset.File.from_files(path=datastore_path)
file_paths = dataset.to_path()
csv_files = [file for file in file_paths if file.endswith('.csv')]

# Read only the first CSV file
first_csv_path = csv_files[0]
#first_csv_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, first_csv_path)])
#df = first_csv_dataset.to_pandas_dataframe()

# Display the DataFrame
#print(df.head())
#first_csv_path

# Check if there are any CSV files and read the first one
if csv_files:
    first_csv_path = csv_files[0]
    
    # Use dataset.download to retrieve the file
    local_path = dataset.download(target_path='.', overwrite=True)
    
    # Read the first CSV file using pandas
    first_csv_df = pd.read_csv(local_path[0])  # local_path[0] refers to the downloaded first CSV file
    
    # Display the first few rows to confirm it's loaded
    print(first_csv_df.head())
else:
    print("No CSV files found.")




subscription_id = 'b88e2867-6b66-4723-88c1-25f2d77b0394'
resource_group = 'az-srotasengine-dev-uks-001'
workspace_name = 'srotasengine'

workspace = Workspace(subscription_id, resource_group, workspace_name)

datastore = Datastore.get(workspace, "workspaceblobstore")
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'UI/2024-11-05_052811_UTC/simulacrum/Data'))
#df = dataset.to_pandas_dataframe()
#df.head()