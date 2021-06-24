import azureml.core
from azureml.core import Workspace, Dataset

from scipy.io import loadmat                # To load .mat files


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

# A dataset is used to reference the data you uploaded to Azure Blob Storage.
# Datasets are an abstraction layer on top of your data that are designed to improve reliability and trustworthiness.
# From the default datastore, we want to extract the uploaded file (.mat) available in the Azure Blob Storage
# The path of the file should be the one available in the Azure Blob Container directory and not the one created locally on Azure Machine Learning > Author > Notebooks
# Dataset.File.from_files() returns a 'FileDataset' object.

#Create a file dataset from the path on the datastore (this may take a short while) for the Ground Truth Maps
files_gt = Dataset.File.from_files(path=(default_ds, 'NEMESIS_images/GroundTruthMaps/*.mat'))

# Download file paths available in the connected Azure Blob Storage. It returns an array with all file paths downloaded locally
arrayDataset_gt = files_gt.download()


# Get the files in the dataset
for gt in arrayDataset_gt:
    gt_mat = loadmat(gt) 
    print(gt_mat['groundTruthMap'].shape[1])
    stop

#Create a file dataset from the path on the datastore (this may take a short while) for the preProcessedImages
files_preProcessed = Dataset.File.from_files(path=(default_ds, 'NEMESIS_images/preProcessedImages/*.mat'))

# Get the files in the dataset
for file_path in files_preProcessed.to_path():
    print(file_path)