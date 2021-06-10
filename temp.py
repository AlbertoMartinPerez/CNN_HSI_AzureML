from scipy.io import loadmat        # Import scipy.io to load .mat files
import numpy as np                  # Import numpy

dir_gtMaps = "NEMESIS_images/GroundTruthMaps/"
dir_preProImages = "NEMESIS_images/preProcessedImages/"

patient = 'ID0018C09'

# Load ground truth map
gt_mat = loadmat(dir_gtMaps + 'SNAPgt' + patient + '_cropped_Pre-processed.mat')
preProcessed_mat = loadmat(dir_preProImages + 'SNAPimages' + patient + '_cropped_Pre-processed.mat')

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