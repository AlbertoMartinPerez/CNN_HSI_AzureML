
#*#####################################################################################################
#* DESCRIPTION OF THIS SCRIPT:
#* Basic script to learn how to use the 'CubeManager' class from 'hsi_dataManager.py' file with Azure
#* Machine Learning and Azure SDK for Python.
#*######################################################################################################

import torch                        # Import PyTorch

import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 
import nn_models as models          # Import 'nn_models.py' file as 'models' to define any new Neural Network included in the file 
import metrics as mts               # Import 'metrics.py' file as 'mts' to evluate metrics

# Import Azure SKD for Python packages
from azureml.core import Run

import os                           # To extract path directory
import joblib                       # To save trained model
import argparse                     # To get all arguments passed to this script if using Azure

#*#############################
#*#### START MAIN PROGRAM #####
#*

# Python dictionary to convert labels to label4Classes
dic_label = {'101': 1, '200': 2, '220': 2, '221': 2, '301': 3, '302': 4, '320': 5}

# Variable to indicate if using data from Azure
useAzure = True

if useAzure:

    # load the diabetes dataset
    print("Loading arguments from the control Script run...")

    # Get script arguments 
    # (file datasets mount points for gt maps and preprocessed cubes)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-data', type=str, dest='data_folder', help='Ground truth map data mount point')
    parser.add_argument('--preProcessed-data', type=str, dest='data_folder', help='Pre-processed cubes data mount point')
    parser.add_argument('--patients_list_train', type=str, dest='patients_list_train', help='List of patients used to train CNN models')
    parser.add_argument('--patient_test', type=str, dest='patient_test', help='List of patients used to classify with trained CNN model')
    parser.add_argument('--batch_dim', type=str, dest='batch_dim', default='3D', help='Batch dimension (3D or 2D)')
    parser.add_argument('--epochs', type=int, dest='epochs', default=100, help='Number of epochs used to train CNN models')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=16, help='Size of batches. Number of patches included in each batch')
    parser.add_argument('--patch_size', type=int, dest='patch_size', default=7, help='Heigh and width size of patches (square patches)')
    parser.add_argument('--k_folds', type=int, dest='k_folds', default=5, help='Number of k-folds to use during double-cross validation')
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001, help='Learning rate parameter')
    parser.add_argument('--model_name', type=str, dest='model_name', default='Conv2DNet_default', help='Name of the CNN model')

    args = parser.parse_args()

    # Load all parameters passed as input to the script
    patients_list_train = [str(patient) for patient in args.patients_list_train.split(',')]
    patient_test = [args.patient_test]

    batch_dim = args.batch_dim
    epochs = args.epochs
    batch_size = args.batch_size
    patch_size = args.patch_size
    k_folds = args.k_folds
    lr = args.learning_rate
    model_name = args.model_name

    # Get the experiment run context
    run = Run.get_context()

    # Get the training data path from the input arguments
    # (they will be used when creating an instance from 'CubeManager' class)
    dir_gtMaps = run.input_datasets['gtMaps_data'] + '/'
    dir_preProImages = run.input_datasets['preProcessed_data'] + '/'

    # Save in log file all defined parameters
    run.log_list('Patients used for training', patients_list_train)
    run.log_list('Patients used for testing', patient_test)
    run.log('Batch dimensions',  batch_dim)
    run.log('Number of epochs',  epochs)
    run.log('Batch size',  batch_size)
    run.log('Patch size', patch_size)
    run.log('Number of K folds', k_folds)
    run.log('Learning rates', lr)

else:
    # Desired patient images ID
    patients_list_train = ['ID0030C02', 'ID0033C02', 'ID0035C02']
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

print("\tTraining batches have been created.")

if ( batch_dim == '2D' ):
    print("\tConverting data and label batches to tensors...")
    # Convert 'data' and 'label4Classes' batches to PyTorch tensors for training our Neural Network
    data_tensor_batch = cm_train.batch_to_tensor(batches_train['data'], data_type = torch.float)
    labels_tensor_batch = cm_train.batch_to_tensor(batches_train['label4Classes'], data_type = torch.LongTensor)

    print("\tTensors have been created.")

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


# If using Azure, log the metrics
if useAzure:
    # Save in log file all obtained metrics
    run.log('OACC', metrics['OACC'])
    run.log_list('ACC',  metrics['ACC'])
    run.log_list('SEN', metrics['SEN'])
    run.log_list('SPE', metrics['SPE'])
    run.log_list('PRECISSION', metrics['PRECISION'])
    run.log('CONFUSION MATRIX', metrics['CON_MAT'])

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
    fig_predMap, fig_GTs = mts.get_classification_map(pred_labels, true_labels, label_coordenates, dims = dims, title="Test classification Map", plot = False, save_plot = False, save_path = None, plot_gt = True, padding=cm_test.pad_margin)

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
    fig_predCube, _ = mts.get_classification_map(pred_labels=pred_labels, true_labels=None, coordenates=cube_coordenates, dims=dims, title="Test Cube classification Map", plot = False, save_plot = False, save_path = None, plot_gt = False, padding=cm_test.pad_margin)


# If using Azure, log classification maps and end run
if useAzure:

    run.log_list('Patients used to classify', patient_test)
    run.log_image(name='Predicted GT classification map', plot=fig_predMap)
    run.log_image(name='Predicted and true GT classification maps', plot=fig_GTs)
    run.log_image(name='Predicted cube classification map', plot=fig_predCube)

    # Save the trained model in the outputs folder
    os.makedirs('outputs', exist_ok=True)
    # Save best PyTorch model
    torch.save(model, './outputs/best_CNN_model.pt')
    # To solve the following error, we have to specifiy that the model will be used on a CPU
    # 'RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. 
    # If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map 
    # your storages to the CPU.'
    PyTorch_model = torch.load('./outputs/best_CNN_model.pt', map_location=torch.device('cpu'))
    joblib.dump(value=PyTorch_model, filename='./outputs/PyTorch_model.pt')

    # Upload the model into the run history record
    # name = The name of the file to upload.
    # path_or_stream = The relative local path or stream to the file to upload.
    run.upload_file(name='./outputs/PyTorch_model.pt', path_or_stream='./outputs/PyTorch_model.pt')

    run.complete()
    
    print('\nAzure run is now completed.')

    # Register the model
    run.register_model(model_path='./outputs/PyTorch_model.pt', model_name=model_name, model_framework='PyTorch', model_framework_version=torch.__version__)


#*#### END MAIN PROGRAM #####
#*###########################