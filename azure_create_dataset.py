import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set_gt = Dataset.File.from_files(path=(default_ds, 'NEMESIS_images/GroundTruthMaps/*.mat'))

# Get the files in the dataset
for file_path in file_data_set_gt.to_path():
    print(file_path)

#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set_preProcessed = Dataset.File.from_files(path=(default_ds, 'NEMESIS_images/preProcessedImages/*.mat'))
# Get the files in the dataset
for file_path in file_data_set_preProcessed.to_path():
    print(file_path)

# Register the file dataset
try:
    file_data_set_gt = file_data_set_gt.register(workspace=ws,
                                            name='GroundTruthMaps',
                                            description='NEMESIS-3D-CM Ground Truth Maps',
                                            tags = {'format':'mat'},
                                            create_new_version=True)
    
    file_data_set_preProcessed = file_data_set_preProcessed.register(workspace=ws,
                                            name='preProcessedImages',
                                            description='NEMESIS-3D-CM preProcessed Images',
                                            tags = {'format':'mat'},
                                            create_new_version=True)
except Exception as ex:
    print(ex)

print('Datasets registered')