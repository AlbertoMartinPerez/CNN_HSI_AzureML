from scipy.io import loadmat        # Import scipy.io to load .mat files
import numpy as np                  # Import numpy
import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 

dir_gtMaps = "NEMESIS_images/GroundTruthMaps/"
dir_preProImages = "NEMESIS_images/preProcessedImages/"

patient = 'ID0018C09'

#*########################################
#* TESTING load_patients_cube() METHOD
#*

# Create an instance of 'DatasetManager'
dm_cubes = hsi_dm.DatasetManager(patch_size = 7, batch_size = 64)

dm_cubes.load_patient_cubes(patients_list = ['ID0018C09'], dir_path_gt = dir_gtMaps, dir_par_preProcessed = dir_preProImages)

#*
#* TESTING load_patients_cube() METHOD
#*########################################

# Load ground truth map
gt_mat = loadmat(dir_gtMaps + 'SNAPgt' + patient + '_cropped_Pre-processed.mat')
preProcessed_mat = loadmat(dir_preProImages + 'SNAPimages' + patient + '_cropped_Pre-processed.mat')


print(dm_cubes.patient_cubes.keys())

if( np.equal(gt_mat['groundTruthMap'].all(), dm_cubes.patient_cubes[patient]['pad_groundTruthMap'].all())):
    print("Shape of padded GT = ", str(dm_cubes.patient_cubes[patient]['pad_groundTruthMap'].shape))
    print("Shape of raw GT = ", str(gt_mat['groundTruthMap'].shape))
    print('GT maps are equal')

stop

# Print keys of the loaded .mat file
print(gt_mat.keys())
print(preProcessed_mat.keys())

# Print shape of entire groundTruthMap
gt_map = gt_mat['groundTruthMap']
print(gt_map.shape)

# Extract coordenates for every label in the ground truth map loaded
for label in np.unique(gt_map)[1::]:
    print('Current label: ' ,str(label))
    x, y = np.nonzero(gt_map == label)

# Print coordenates
print((x, y))