import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get the default datastore
default_ds = ws.get_default_datastore()

# Enumerate all datastores, indicating which is the default
for ds_name in ws.datastores:
    print(ds_name, "- Default =", ds_name == default_ds.name)


# Desired patient images ID
# ['ID0018C09', 'ID0025C02', 'ID0029C02', 'ID0030C02', 'ID0033C02', 'ID0034C02', 'ID0035C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02', 'ID0064C04',
# 'ID0064C06', 'ID0065C01', 'ID0065C09', 'ID0067C01', 'ID0068C08', 'ID0070C02', 'ID0070C05', 'ID0070C08', 'ID0071C02', 'ID0071C011', 'ID0071C014']
patients_list = ['ID0018C09', 'ID0025C02', 'ID0029C02', 'ID0030C02', 'ID0033C02', 'ID0034C02', 'ID0035C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02']

# Upload files to the default datastore 'default_ds'
for patient in patients_list:
    # Upload GroundTruthMaps
    gt_file = './NEMESIS_images/GroundTruthMaps/SNAPgt' + patient + '_cropped_Pre-processed.mat'
    default_ds.upload_files(files=[gt_file],
                        target_path='NEMESIS_images/GroundTruthMaps/', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)

    # Upload preProcessedImages
    preProcessed_file = './NEMESIS_images/preProcessedImages/SNAPimages' + patient + '_cropped_Pre-processed.mat'
    default_ds.upload_files(files=[preProcessed_file], # Upload the diabetes csv files in /data
                        target_path='NEMESIS_images/preProcessedImages/', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)