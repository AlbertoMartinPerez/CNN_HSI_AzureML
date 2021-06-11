from scipy.io import loadmat        # Import scipy.io to load .mat files
import numpy as np                  # Import numpy
import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 

dir_gtMaps = "NEMESIS_images/GroundTruthMaps/"
dir_preProImages = "NEMESIS_images/preProcessedImages/"
dir_datasets = "NEMESIS_images/datasets/"

patient = 'ID0018C09'

patient_list = ['ID0018C09', 'ID0025C02'] #, 'ID0029C02', 'ID0030C02', 'ID0033C02', 'ID0034C02', 'ID0035C02', 'ID0038C02', 'ID0047C02', 'ID0047C08', 'ID0050C05', 'ID0051C05', 'ID0056C02']

#*########################################
#* TESTING load_patients_cube() METHOD
#*

# Python dictionary to convert labels to label4Classes
dic_label = {'101': 1, '200': 2, '220': 2, '221': 2, '301': 3, '302': 4, '320': 5, '331': 6}

# Create an instance of 'DatasetManager'
dm_cubes = hsi_dm.DatasetManager(patch_size = 7, batch_size = 64, dic_label = dic_label)

dm_cubes.load_patient_cubes(patients_list = patient_list, dir_path_gt = dir_gtMaps, dir_par_preProcessed = dir_preProImages)

print('\ncubes_data: ')
print('\tShape of cubes_data: ', str(dm_cubes.cubes_data.shape))
print('\tFirst row: ', str(dm_cubes.cubes_data[0, :]))

print('\ncubes_label: ')
print('\tShape of cubes_label: ', str(dm_cubes.cubes_label.shape))
print('\tFirst row: ', str(dm_cubes.cubes_label[0, :]))

print('\ncubes_label4Classes: ')
print('\tShape of cubes_label4Classes: ', str(dm_cubes.cubes_label4Classes.shape))
print('\tFirst row: ', str(dm_cubes.cubes_label4Classes[0, :]))

for patient in patient_list:
    print('\nPatient: ', str(patient))
    print('\tShape: ', str(dm_cubes.patient_cubes[str(patient)]['label_coords'].shape))
    print('\tFirst row: ', str(dm_cubes.patient_cubes[str(patient)]['label_coords'][0, :]))

stop

#*
#* TESTING load_patients_cube() METHOD
#*########################################

for patient in patient_list:
    print(np.unique(dm_cubes.patient_cubes[patient]['raw_groundTruthMap'])[1::])

stop

print(dm_cubes.patient_cubes.keys())


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