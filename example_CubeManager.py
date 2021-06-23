######################################################################################################
# DESCRIPTION OF THIS SCRIPT:
# Basic script to learn how to use the 'CubeManager' class from 'hsi_dataManager.py' file.
#-----------------------------------------------------------------------------------------------------
# todo: UPDATE THIS LIST!
# It demonstrates:
#   1) How to create a CubeManager instance
#   2) How to load '_dataset.mat' files to the CubeManager instance for training
#   3) How to create 2D batches with loaded data in the CubeManager instance
#   4) How to convert 2D batches to batch PyTorch tensors with the CubeManager instance
#   5) How to create a FourLayerNet model and how to train it with the batch 
#   6) How to load a single '_dataset.mat' with a new CubeManager instance for testing
#   7) How to predict the test dataset with our trained model
#   8) How to compute prediction metrics and print them
#######################################################################################################

import torch                        # Import PyTorch

import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 
import nn_models as models          # Import 'nn_models.py' file as 'models' to define any new Neural Network included in the file 
import metrics as mts               # Import 'metrics.py' file as 'mts' to evluate metrics

#*#############################
#*#### START MAIN PROGRAM #####
#*

# Desired patient images ID
# ['ID0018C09', 'ID0025C02', 'ID0029C02', 'ID0030C02', 'ID0033C02', 'ID0034C02', 'ID0035C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02',
# 'ID0064C04', 'ID0064C06', 'ID0065C01', 'ID0065C09', 'ID0067C01', 'ID0068C08', 'ID0070C02', 'ID0070C05', 'ID0070C08', 'ID0071C02', 'ID0071C011', 'ID0071C014']
patients_list_train = ['ID0030C02', 'ID0033C02', 'ID0035C02']#, 'ID0034C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02']
patient_test = ['ID0033C02']

# Directories with data
dir_datasets = "NEMESIS_images/datasets/"
dir_gtMaps = "NEMESIS_images/GroundTruthMaps/"
dir_preProImages = "NEMESIS_images/preProcessedImages/"
dir_rawImages = "NEMESIS_images/tif/"

# Python dictionary to convert labels to label4Classes
dic_label = {'101': 1, '200': 2, '220': 2, '221': 2, '301': 3, '302': 4, '320': 5}

# Determine dimension of batches for the Neural Network
batch_dim = '3D'

# Number of epochs
epochs = 1

# Batch size
batch_size = 16

# Patch size (recommended to be always odd)
patch_size = 7

# K_folds
k_folds = 2

# Learning rate
lr = 0.01

#*####################
#* LOAD TRAIN IMAGES
print("\n##########")
print("Loading training images. Please wait...")

# Create an instance of 'CubeManager'
cm_train = hsi_dm.CubeManager(patch_size = patch_size, batch_size = batch_size, dic_label = dic_label, batch_dim = batch_dim)

# Load all desired pixels to the 'CubeManager' instance 'cm_train' (all data is stored inside the instance attributes)
cm_train.load_patient_cubes(patients_list_train, dir_gtMaps, dir_preProImages)

print("\tTraining images have been loaded. Creating training batches...")

# Create batches with the loaded data. Returns 'batches' which is a Python dictionary including 2 Python lists, 'data' and 'labels', containing all batches
batches_train = cm_train.create_batches()

"""
# PRINT IN TERMINAL THE SHAPE OF EVERY CREATED BATCH
if ( batch_dim == '2D' ):
    i = 0
    for b in batches_train['data']:
        print('Size of batch ', i+1, ' = ', batches_train['data'][i].shape )
        i += 1

    print('Last batch ', batches_train['data'][i-1] )

elif ( batch_dim == '3D' ):
    i = 0
    for b in batches_train['cube']:
        print('Size of batch ', i+1, ' = ', batches_train['cube'][i].shape )
        i += 1

    print('Last batch ', batches_train['cube'][i-1] )

stop
"""

"""
print("\n\t#####")
print('\t batches_train:')
print("\t\t type(batches_train['cube']) = ", type(batches_train['cube']))
print("\t\t len(batches_train['cube'] = ", len(batches_train['cube']))
print("\t\t type(batches_train['cube'][0]) = ", type(batches_train['cube'][0]))
print("\t\t batches_train['cube'][0].shape = ", batches_train['cube'][0].shape )
print("\t\t batches_train['cube'][0][0].shape = ", batches_train['cube'][0][0].shape )
"""


print("\tTraining batches have been created.")

if ( batch_dim == '2D' ):
    print("\tConverting data and label batches to tensors...")
    # Convert 'data' and 'label4Classes' batches to PyTorch tensors for training our Neural Network
    data_tensor_batch = cm_train.batch_to_tensor(batches_train['data'], data_type = torch.float)
    labels_tensor_batch = cm_train.batch_to_tensor(batches_train['label4Classes'], data_type = torch.LongTensor)

    print("\tTensors have been created.")

"""    
    print('### DEBUG ###')
    print('data_tensor_batch: ')
    print('\t type(data_tensor_batch) = ', type(data_tensor_batch))
    print('\t data_tensor_batch[0].shape = ', data_tensor_batch[0].shape)
    print('\t type(labels_tensor_batch) = ', type(labels_tensor_batch))
    print('\t labels_tensor_batch[0].shape = ', labels_tensor_batch[0].shape)
 """   

#*######################
#* TRAIN NEURAL NETWORK
print("\n##########")
print("Training your Neural Network. Please wait...")

if ( batch_dim == '2D' ):
    # Create a FourLayerNet model, which contains 4 fully connected layers with relu activation functions
    model = models.FourLayerNet(D_in = cm_train.data.shape[-1], H = 16, D_out = cm_train.numUniqueLabels)

    # Train FourLayerNet model
    model.trainNet(batch_x = data_tensor_batch, batch_y = labels_tensor_batch, epochs = epochs, plot = True, lr = lr)

elif ( batch_dim == '3D' ):

    # Create a CrossValidator instance
    cv = hsi_dm.CrossValidator(batch_data=batches_train['cube'], batch_labels=batches_train['label'], k_folds=k_folds, numUniqueLabels=cm_train.numUniqueLabels, numBands=cm_train.numBands, epochs=epochs, lr=lr)

    # Perform K-fold double-cross validation
    cv.double_cross_validation()

    # Save in 'model' the best model obtained from the double-cross validation
    model = cv.bestModel

#*###################
#* LOAD TEST IMAGES
print("\n##########")
print("Loading test image. Please wait...")

# Create an instance of 'CubeManager'
cm_test = hsi_dm.CubeManager(patch_size = patch_size, batch_size = batch_size, dic_label = dic_label, batch_dim = batch_dim)

# Load all desired pixels to the 'CubeManager' instance 'cm_test' (all data is stored inside the instance attributes)
cm_test.load_patient_cubes(patients_list = patient_test, dir_path_gt = dir_gtMaps, dir_par_preProcessed = dir_preProImages)

print("\tTest image has been loaded. Creating test batches...")

# Create batches with the loaded data. Returns 'batches' which is a Python dictionary including 2 Python lists, 'data' and 'labels', containing all batches
batches_test = cm_test.create_batches()

print("\tTest batches have been created. Converting data batches to tensors...")

if ( batch_dim == '2D' ):
    # Convert 'data' batches to PyTorch tensors for training our Neural Network
    data_tensor_batch_test = cm_test.batch_to_tensor(batches_test['data'], data_type = torch.float)
elif ( batch_dim == '3D' ):
    # Convert 'cube' batches to PyTorch tensors for training our Neural Network
    data_tensor_batch_test = cm_test.batch_to_tensor(batches_test['cube'], data_type = torch.float)

print("\tTensors have been created.")

#*##############################################
#* PREDICT TEST IMAGES WITH OUT NEURAL NETWORK
print("\n##########")
print("Predict loaded test image with trained model.")
print("\nModel predicting patient image = ", cm_test.patients_list[0] )

if ( batch_dim == '2D' ):
    # Predict with the FourLayerNet model
    pred_labels = model.predict(batch_x = data_tensor_batch_test)
elif ( batch_dim == '3D' ):
    # Predict with the Conv2DNet model
    pred_labels = model.predict(batch_x = data_tensor_batch_test)

#*##############################################
#* COMPUTE METRICS WITH THE MODEL PREDICTION

if ( batch_dim == '2D' ):
    # Evaluate how well the model can predict a new image unused during training
    # batches['label4Classes']: is a Python list where each element contains the labels for each of the samples in the corresponding batch
    # by calling the 'batch_to_label_vector()' method, we generate a column numpy array from the Python list and store all batches labels in order
    # pred_labels: is a numpy column vector with all predicted labels of all batches in order
    metrics = mts.get_metrics(cm_test.batch_to_label_vector(batches_test['label4Classes']), pred_labels, cm_test.numUniqueLabels)

elif ( batch_dim == '3D' ):
    # 'batches_test['label']' contains (x_coord, y_coord, labels). We first convert this Python list to a label vector.
    # Then we need to extract all labels by using '[:, -1]'. This gives a (N,) vector, but we need to make it (N,1) to
    # compare it with the predicted labels. Also, a conversion to 'int' is needed so 'get_metrics' works properly.
    metrics = mts.get_metrics(cm_test.batch_to_label_vector(batches_test['label'])[:, -1].reshape((-1,1)).astype(int), pred_labels, cm_test.numUniqueLabels)


print("\nMetrics after predicting:")
print('\tOACC = ', str(metrics['OACC']))
print('\tACC = ', str(metrics['ACC']))
print('\tSEN = ', str(metrics['SEN']))
print('\tSPE = ', str(metrics['SPE']))
print('\tPRECISION = ', str(metrics['PRECISION']))
print('\tCONFUSION MATRIX: \n\t', str(metrics['CON_MAT']))

#*###############################
#* COMPUTE CLASSIFICATION MAP

print("\n##########")
print("Plotting classification maps")

# To compute classification maps, it is necessary to have used the 'CubeManager' class, since it
# provides the X and Y coordenates for every pixel in every predicted batch.
# Please note that 'DataManager' class does no provide the coordenates to any sample.

if ( batch_dim == '2D' ):
    # Concatenate all list elements from 'batches_test['label4Classes']' (all label batches) to a numpy array
    true_labels = cm_test.concatenate_list_to_numpy(batches_test['label4Classes'])
    # Do the same with the coordenates to know the predicted label and its corresponding position
    label_coordenates = cm_test.concatenate_list_to_numpy(batches_test['label_coords']).astype(int)

    # Extract dimension of the loaded groundTruthMap for the test patient
    dims = cm_test.patient_cubes[patient_test[0]]['raw_groundTruthMap'].shape

    # Generate classification map from the predicted labels
    mts.get_classification_map(pred_labels, true_labels, label_coordenates, dims, title="Test classification Map", plot = True, save_plot = False, save_path = None, plot_gt = False)

if ( batch_dim == '3D' ):
    # Concatenate all list elements from 'batches_test['label']' (all label batches) to a numpy array
    true_labels = cm_test.concatenate_list_to_numpy(batches_test['label'])[:, -1].reshape((-1,1)).astype(int)
    # Do the same with the coordenates to know the predicted label and its corresponding position
    label_coordenates = cm_test.concatenate_list_to_numpy(batches_test['label'])[:, 0:-1].astype(int)

    # Extract dimension of the loaded groundTruthMap for the test patient
    dims = cm_test.patient_cubes[patient_test[0]]['pad_groundTruthMap'].shape

    # Generate classification map from the predicted labels
    mts.get_classification_map(pred_labels, true_labels, label_coordenates, dims = dims, title="Test classification Map", plot = True, save_plot = False, save_path = None, plot_gt = True, padding=cm_test.pad_margin)

#*######################################################
#* PREDICT WITH THE MODEL THE ENTIRE PREPROCESSED CUBE

print("\n##########")
print("Predicting the entire preProcessed cube...")

if ( batch_dim == '3D' ):
    # Generate batches for the entire preProcessed cube
    cube_batch = cm_test.create_cube_batch()

    # Convert 'cube' batches to PyTorch tensors for training our Neural Network
    cube_tensor_batch = cm_test.batch_to_tensor(cube_batch['data'], data_type = torch.float)

    # Obtain 'cube' batches coordenates
    cube_coordenates = cm_test.concatenate_list_to_numpy(cube_batch['coords']).astype(int)

    # Predict with the Conv2DNet model
    pred_labels = model.predict(batch_x = cube_tensor_batch)

    # Generate classification map from the predicted labels
    mts.get_classification_map(pred_labels=pred_labels, true_labels=None, coordenates=cube_coordenates, dims=dims, title="Test Cube classification Map", plot = True, save_plot = False, save_path = None, plot_gt = False, padding=cm_test.pad_margin)

#*#### END MAIN PROGRAM #####
#*###########################