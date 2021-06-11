#################################################################################################################
# DESCRIPTION OF THIS SCRIPT:
# The idea is to prepare locally all scripts and stop wasting Azure credits inside Azure Machine Learning
#################################################################################################################

import torch                        # Import PyTorch

import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 
import nn_models as models          # Import 'nn_models.py' file as 'models' to define any new Neural Network included in the file 
import metrics as mts               # Import 'metrics.py' file as 'mts' to evluate metrics

#*#############################
#*#### START MAIN PROGRAM #####
#*

# Desired patient images ID
# ['ID0018C09', 'ID0025C02', 'ID0029C02', 'ID0030C02', 'ID0033C02', 'ID0034C02', 'ID0035C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02', 'ID0064C04',
# 'ID0064C06', 'ID0065C01', 'ID0065C09', 'ID0067C01', 'ID0068C08', 'ID0070C02', 'ID0070C05', 'ID0070C08', 'ID0071C02', 'ID0071C011', 'ID0071C014']
patients_list_train = ['ID0018C09', 'ID0025C02']

# Directories with data
dir_datasets = "NEMESIS_images/datasets/"
dir_gtMaps = "NEMESIS_images/GroundTruthMaps/"
dir_preProImages = "NEMESIS_images/preProcessedImages/"
dir_rawImages = "NEMESIS_images/tif/"

# Python dictionary to convert labels to label4Classes
dic_label = {'101': 1, '200': 2, '220': 2, '221': 2, '301': 3, '302': 4, '320': 5, '331': 6}

#*####################
#* LOAD TRAIN IMAGES

# Create an instance of 'DatasetManager'
dm_train = hsi_dm.DatasetManager(patch_size = 7, batch_size = 64)

# Load all desired pixels to the 'DatasetManager' instance 'dm_train' (all data is stored inside the instance attributes)
dm_train.load_patient_datasets(patients_list = patients_list_train, dir_path = dir_datasets)

# Create batches with the loaded data. Returns 'batches' which is a Python dictionary including 2 Python lists, 'data' and 'labels', containing all batches
batches_train = dm_train.create_2d_batches()

"""
# PRINT IN TERMINAL THE SHAPE OF EVERY CREATED BATCH
i = 0
for b in batches['data']:
    print('Size of batch ', str(i+1), ' = ', str(batches['data'][i].shape))
    i += 1
"""

# Convert 'data' and 'labels' batches to PyTorch tensors for training our Neural Network
data_tensor_batch = dm_train.batch_to_tensor(batches_train['data'], data_type = torch.float)
labels_tensor_batch = dm_train.batch_to_tensor(batches_train['label4Classes'], data_type = torch.LongTensor)


#*######################
#* TRAIN NEURAL NETWORK

# Create a FourLayerNet model, which contains 4 fully connected layers with relu activation functions
model = models.FourLayerNet(D_in = dm_train.data.shape[-1], H = 16, D_out = dm_train.dataset_numUniqueLabels)

# Train FourLayerNet model
model.trainNet(batch_x = data_tensor_batch, batch_y = labels_tensor_batch, epochs = 10, plot = True, lr = 0.01)


#*###################
#* LOAD TEST IMAGES

# Create an instance of 'DatasetManager'
dm_test = hsi_dm.DatasetManager(patch_size = 7, batch_size = 64)

# Load all desired pixels to the 'DatasetManager' instance 'dm_test' (all data is stored inside the instance attributes)
dm_test.load_patient_datasets(patients_list = ['ID0038C02'], dir_path = dir_datasets)

# Create batches with the loaded data. Returns 'batches' which is a Python dictionary including 2 Python lists, 'data' and 'labels', containing all batches
batches_test = dm_test.create_2d_batches()

# Convert 'data' batches to PyTorch tensors for testing our Neural Network
data_tensor_batch_test = dm_test.batch_to_tensor(batches_test['data'], data_type = torch.float)


#*##############################################
#* PREDICT TEST IMAGES WITH OUT NEURAL NETWORK

# Predict with the FourLayerNet model
print("\nModel predicting patient image = ", str(dm_test.patients_dataset_list[0]))
pred_labels = model.predict_2d(batch_x = data_tensor_batch_test)

# Evaluate how well the model can predict a new image unused during training
# batches['label4Classes']: is a Python list where each element contains the labels for each of the samples in the corresponding batch
# by calling the 'batch_to_label_vector()' method, we generate a column numpy array from the Python list and store all batches labels in order
# pred_labels: is a numpy column vector with all predicted labels of all batches in order
metrics = mts.get_metrics(dm_test.batch_to_label_vector(batches_test['label4Classes']), pred_labels, dm_test.dataset_numUniqueLabels)

print("\nMetrics after predicting:")
print('\tOACC = ', str(metrics['OACC']))
print('\tACC = ', str(metrics['ACC']))
print('\tSEN = ', str(metrics['SEN']))
print('\tSPE = ', str(metrics['SPE']))
print('\tPRECISION = ', str(metrics['PRECISION']))
print('\tCONFUSION MATRIX: \n\t', str(metrics['CON_MAT']))


#*#### END MAIN PROGRAM #####
#*###########################