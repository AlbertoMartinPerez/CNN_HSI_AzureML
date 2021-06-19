#################################################################################
# Alberto Martín Pérez - 2021
# This concept has been extracted from Guillermo Vázquez Valle dataset_manager.py
#--------------------------------------------------------------------------------
# This script is used to manipulate HSI data and create small patches or batches.
# It uses the groundtruth pixel references as a mask and extract all HS pixels 
# from around them to create patches. It can also use already made .mat datasets.
#################################################################################

import numpy as np                          # Import numpy
import torch                                # Import PyTorch
from scipy.io import loadmat                # Import scipy.io to load .mat files
from sklearn.model_selection import KFold   # Import KFold cross-validator from sklearn

import metrics as mts                       # Import 'metrics.py' file as 'mts' to evluate metrics inside CrossValidator class
import nn_models as models                  # Import 'nn_models.py' file as 'models' to define any new Neural Network included in the file 

#*################################
#*#### DatasetManager class  #####
#*
class DatasetManager:
    """
    This class is used to load '_dataset.mat' files destined to work with Neural Network models, both for training and classification.
    - Recomendation: Create 2 instances of DatasetManager, 1 to load all data to train and another 1 to load only the image to classify.
                     This way, in this second instance we can predict a new patient.
    - Note 1: 'DatasetManager' only works with numpy arrays!
    - Note 2: 'DatasetManager' cannot work with pixel coordenates, since '_dataset.mat' files do not store them. To work with coordenates, please use 'CubeManager' class.
    
    """
    #*##########################
    #*#### DEFINED METHODS #####
    #*

    def __init__(self, batch_size = 64):
        """
        Define the constructor of 'DatasetManager' class. It only works with '_dataset.mat' files.

        Inputs
        ----------
        - 'batch_size': Integer. Size of each data batch.

        Attributes
        ----------
        - 'patients_list':      Python list. Attribute to store all patient IDs as a python list with strings. Used when _datasets.mat files are loaded.
        - 'data':               Numpy array (but initialized as Python list). Attribute to store all dataset pixels once all patients have been loaded. All data will be appended.
        - 'label':              Numpy array (but initialized as Python list). Attribute to store all dataset labels once all patients have been loaded. All labels will be appended.
        - 'label4Classes':      Numpy array (but initialized as Python list): Attribute to store all dataset labels 4 classes once all patients have been loaded. All label4Classes will be appended.
        - 'numUniqueLabels':    Integer. Attribute to store the total number of different labels once all patients have been loaded.
        - 'numTotalSamples':    Integer. Attribute to store the total number of samples once all patients have been loaded.

        """
        #* CREATE ATTRIBUTES FOR THE INSTANCES
        # Global attributes
        self.batch_size = batch_size

        #* Attributes related to '_dataset.mat' files 
        self.patients_list = []
        self.data = []
        self.label = []
        self.label4Classes = []

        self.numUniqueLabels = None
        self.numTotalSamples = None

    def __largest_class(self):
        """
        (Private method) Look for the labeled class with more elements from a numpy vector.
        Important: This method works if a call to 'load_patient_datasets()' or
        'load_patient_cubes()' was made first.

        Inputs
        ----------
        - 'data_flag': String indicating if we want to evaluate 'datasets' or 'cubes'.
        
        Outputs
        ----------
        - Integer number indicating the label with largest number of elements
        """
        # Create empty array with the same dimensions as the number of unique labels
        temp_labels = np.zeros(self.numUniqueLabels, dtype=int)

        #*#########################################################
        #* FOR LOOP ITERATES OVER ALL AVAILABLE LABELS FROM THE
        #* LOADED OR GENERATED DATASET.
        #*
        for label in np.unique(self.label4Classes)[0::]:
            # Store the number of pixels that are 'label'
            temp_labels[label-1] = np.count_nonzero(self.label4Classes == label)
        #*
        #* END FOR LOOP
        #*##############

        return (np.where(temp_labels == np.amax(temp_labels))[0] + 1)   # Return the label containing the largest amount of elements (+1 since np.where returns the index starting at 0)

    def load_patient_datasets(self, patients_list, dir_path):
        """
        Load all patient '.mat' datasets from the input list 'patients_list'. It saves the data in 2 'DatasetManager' attributes, 'self.data' and 'self.label4Classes'. 
        If more than 1 patient is given in the list, the data is appended to those attributes, so that each index in the attribute corresponds to 1 single patient.
        It also stores the python 'patients_list' as a 'DatasetManager' attribute, so that we know which patients have been used.
        - Important: '_dataset.mat' files need to have 'data', 'label' and 'label4Classes' name fields.

        Inputs
        ----------
        - 'patients_list': Python list including the strings ID for each patient
        - 'dir_path': String that includes the path directory where the files are
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if python list is empty
        if (len(patients_list) == 0):
            raise RuntimeError("Not expected an empty python list input. 'patients_list' is empty.")
        # Check if first python list element is a string
        if not ( isinstance(patients_list[0], str) ):
            raise TypeError("Expected first element of 'patients_list' to be string. Received instead element of type: ", str(type(patients_list[0])) )
        # Once the first element of the list is a string, check if all the elements in the 'patients_list' are also string
        if (len(patients_list) != 1):
            if ( all(element == patients_list[0] for element in patients_list) ):
                raise TypeError("Expected 'patients_list' to only contain string elements. Please ensure all elements in the list are of type 'str' ")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        self.patients_list = patients_list              # Save in the 'self.patients_list' attribute all patient IDs as a python list with strings

        # Load the first patient image to extract
        # Create temporary numpy array to store all samples with 1 row and as many columns as features in the first dataset patient
        temp_data_array = np.zeros((1, loadmat(dir_path + patients_list[0] + '_dataset.mat')['data'].shape[-1]))
        # Create temporary numpy array to store all labels and label4Classes with 1 row and 1 column 
        temp_label_array = np.zeros((1, 1))
        temp_label4Classes_array = np.copy(temp_label_array)

        # Flag to indicate that we have deleted the first row of the temporary array,
        # otherwise we would append always the empty row.
        deleted_row = False

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PATIENTS IN THE INPUT LIST.
        #* IT ALSO APPENDS ALL DATA AND LABELS IN 'self.data' AND 
        #* 'self.label4Classes' ATTRIBUTES
        #*
        for patient in patients_list:

            dataset = loadmat(dir_path + patient + '_dataset.mat')                                          # Load dataset from the current patient

            temp_data_array = np.vstack((temp_data_array, dataset['data']))                                 # Concatenate all samples in 'data' from the current patient
            temp_label_array = np.vstack((temp_label_array, dataset['label']))                              # Concatenate all labels in 'label' from the current patient
            temp_label4Classes_array = np.vstack((temp_label4Classes_array, dataset['label4Classes']))      # Concatenate all labels in 'label' from the current patient

            #*############################################################
            #* IF STATEMENT TO CHECK IF FIRST ROW OF THE TEMP ARRAY HAS
            #* BEEN DELETED. (remember they where created with the first 
            #* row as empty)
            #*
            if not (deleted_row):
                # Delete the first empty row
                temp_data_array = np.delete(temp_data_array, 0, axis = 0)
                temp_label_array = np.delete(temp_label_array, 0, axis = 0)
                temp_label4Classes_array = np.delete(temp_label4Classes_array, 0, axis = 0)
                # Update flag to True
                deleted_row = True
            #*
            #* END OF IF
            #*############

        #*
        #* END FOR LOOP
        #*##############

        # Store in 'self.data', 'self.label' and 'self.label4Classes' instance attributes all loaded data
        
        self.data = temp_data_array
        self.label = temp_label_array.astype(int)
        self.label4Classes = temp_label4Classes_array.astype(int)

        self.numUniqueLabels = len(np.unique(self.label))       # Store in the 'self.numUniqueLabels' attribute a numpy array with the number of unique classes from all stored labels
        self.numTotalSamples = self.data.shape[0]               # Store in the 'self.numTotalSamples' attribute the total number of loaded samples

    def create_batches(self):
        """
        Create a Python dictionary with small batches of size 'batch_size' from the loaded data and their labels. It follows the Random Stratified Sampling methodology.
        Not all batches will be perfectly distributed, since classes with fewer samples may not appear in all batches. Also, in case a batch is not going to comply with
        the input 'batch_size', we add more pixels from the class with more samples.
        Note: If we have few samples left but they are not as big as 'batch_size', then we discard those pixels. (batches need to should be the same size for training)
        Important: This method works when '_dataset.mat' files have been loaded! Therefore, no coordenates are stored.
        
        Outputs
        ----------
        - Python dictionary with 2 Python lists: (they are all in order, so index 0 of any key value would have information of the same sample)
            - A) key = 'data'. Includes 'list_samples': Python list with sample of batches
            - B) key = 'label4Classes'. Includes 'list_labels': Python list with the labels of all batches in 'list_samples'
        """
        #*################
        #* ERROR CHECKER
        #*
        # Check if 'data' instance attribute contains samples. If not, it would mean that no '_dataset.mat' file has been loaded.
        if ( len(self.data) == 0):
            raise RuntimeError("No '_dataset.mat' file has been loaded. To use 'create_2d_batches_from_dataset()' method, please first load datasets using the 'load_patient_datasets()' method.")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        largest_label = self.__largest_class()                          # Find the label with more elements

        data_temp = np.copy(self.data)                                  # Create a copy of the loaded data in a temporary variable. This way, we don't delete data from the instance 'data' attribute.
        label4Classes_temp = np.copy(self.label4Classes)                # Create a copy of the loaded label4Classes in a temporary variable. This way, we don't delete data from the instance 'label4Classes' attribute.

        list_sample_batches = []
        list_label_batches = []

        #*###############################################
        #* WHILE LOOP CREATES 1 BATCH EVERY ITERATION
        #*
        while data_temp.shape[0] >= self.batch_size:                # Stay in this while loop if the number of samples left in 'dataTemp' is equal or greater than 'bath_size'

            list_samples = []                                       # Create empty Python list to append data samples
            list_labels = []                                        # Create empty Python list to append data labels
            size_current_batch = 0

            num_total_samples_left =  data_temp.shape[0]
            
            #*#################################################
            #* FOR LOOP ITERATES OVER EVERY LABEL AVAILABLE
            #*
            for label in np.unique(label4Classes_temp)[0::]:             # Iterate over all labels available

                # [0::] indicates the following thing:
                #	- '0': start a the first element of the list
                #	- ':': end at the last element
                #	- ':': jump in invervals of 1
                class_indices = np.where(label4Classes_temp == label)[0]        # Extract the 'indices' where pixels are labeled as the input 'label'. [0] is to extract the indices, since np.where() returns 2 arrays.
                                                                                # 'class_indices' is an array where each position indicates the index of every labeled pixel equal to 'label'
                
                #*###############################################################################
                #* IF STATEMENT IS TO EVALUATE IF THERE ARE SAMPLES LEFT FOR THE CURRENT LABEL
                #*
                if ( len(class_indices) > 0 ):

                    percentage =  (len(class_indices) / num_total_samples_left)     # Calculate percentage that correspond to the pixel 'labels' out of all samples left unused

                    num_samples = int(round(self.batch_size * percentage))          # Calculate the number of samples to add to the batch for the current label

                    if num_samples == 0: num_samples = 1
                    
                    #*##############################################################
                    #* IF STATEMENT IS TO EVALUATE WHETHER OR NOT I NEED TO
                    #*  SUBSTRACT ANY SAMPLE BECAUSE THE BATCH HAS REACH ITS LIMIT
                    #*
                    if ( (size_current_batch + num_samples) > self.batch_size ):
                        samples_to_substract = ((size_current_batch + num_samples) - self.batch_size)   # Calculate the samples that need to be substracted
                        num_samples -= samples_to_substract                                             # Update the 'num_samples' variable to comply with the 'batch_dimension' size
                        class_indices = class_indices[0:-samples_to_substract]                          # Delete the latest samples that we are going to substract
                    #*
                    #* END OF IF
                    #*############

                    size_current_batch += num_samples                                                               # Update 'size_current_batch' variable to know the size of the current batch

                    sample_indices = np.random.choice(len(class_indices), num_samples, replace=False)               # Randomly select a total of 'num_samples' sample indices for the batch

                    list_samples.append(data_temp[class_indices[sample_indices]])                                    # Store in the Python list all randomly selected samples from the current label
                    list_labels.append(label4Classes_temp[class_indices[sample_indices]])                            # Store in the Python list all randomly selected sample labels from the current label

                    data_temp = np.delete(data_temp, class_indices[sample_indices], axis = 0)                        # Remove the sampled pixels from the temporary 'data' variable
                    label4Classes_temp = np.delete(label4Classes_temp, class_indices[sample_indices], axis = 0)      # Remove the sampled labels from the temporary 'label4Classes' variable   
                #*
                #* END OF IF
                #*############
            #*
            #* END OF FOR LOOP
            #*###################

            # Convert both Python lists 'list_samples' and 'list_labels' to numpy arrays by using 'np.concatenate()'.
            # This way we concatenate all pixels from all labels to be stored in 1 single variable, which would represent 1 single batch
            single_sample_batch = np.concatenate(list_samples, axis=0)
            single_label_batch = np.concatenate(list_labels, axis=0)

            #*##################################################################
            #* IF STATEMENT IS TO ADD ADDITIONAL SAMPLES TO THE CURRENT BATCH 
            #* IN CASE ITS LENGHT IS LESS THAN THE ACTUAL BATCH_SIZE
            #* (We take samples from the label with more classes)
            #*
            if (len(single_sample_batch) < self.batch_size):
                # Calculate the number of elements that we need to add to the batch. Will use the 'label' with more elements
                samples_to_add = (self.batch_size - len(single_sample_batch))

                # Extract the 'indices' where pixels are labeled as the 'largest label'. [0] is to extract the indices, since np.where() returns 2 arrays.
                class_indices_temp = np.where(label4Classes_temp == largest_label)[0]

                # Randomly select a total of 'num_samples' sample indices for the batch
                sample_indices_temp = np.random.choice(len(class_indices_temp), samples_to_add, replace=False)

                # Store in the Python list all randomly selected samples and labels from the current label and the additional samples from the 'largest label' 
                single_sample_batch = np.vstack([single_sample_batch, data_temp[class_indices_temp[sample_indices_temp]]])
                single_label_batch = np.vstack([single_label_batch, label4Classes_temp[class_indices_temp[sample_indices_temp]]])

                # Remove the additional sampled pixels and labels from the temporary 'data' and 'label4Classes variable
                data_temp = np.delete(data_temp, class_indices_temp[sample_indices_temp], axis = 0)
                label4Classes_temp = np.delete(label4Classes_temp, class_indices_temp[sample_indices_temp], axis = 0)
            #*   
            #* END OF IF
            #*##############

            list_sample_batches.append(single_sample_batch)
            list_label_batches.append(single_label_batch)
        #*
        #* END OF WHILE 
        #*################
        
        #*########################################################
        #* IF STATEMENT IS USED TO APPEND THE REMAINING DATA 
        #* THAT CAN NOT BE USED AS A BATCH OF 'batch_size' SIZE
        if( (data_temp.shape[0] / self.batch_size) > 0):
            list_sample_batches.append(data_temp)
            list_label_batches.append(label4Classes_temp)
        #*   
        #* END OF IF
        #*##############

        return {'data':list_sample_batches, 'label4Classes':list_label_batches}

    def batch_to_tensor(self, python_list, data_type):
        """
        Convert all numpy array batches included in a Python list to desired PyTorch tensors types.
        
        Inputs
        ----------
        - 'python_list':    Python list with batches as numpy arrays
        - 'data_type':      PyTorch tensor type to convert the numpy array batch to desired tensor type

        Outputs
        ----------
        - 'tensor_batch':   Python list with batches as PyTorch tensors
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list, list) ):
            raise TypeError("Expected list as input. Received input of type: ", str(type(python_list)) )
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list[0], np.ndarray) ):
            raise TypeError("Expected numpy arrays as input. Received input of type: ", str(type(python_list[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        tensor_batch = []                               # Create empty Python list to return

        #*###############################################################
        #* FOR LOOP TO ITERATE OVER ALL BATCHES INCLUDED IN THE INPUT 
        #* PARAMETER 'python_list' AND CONVERT THEM AS PYTORCH TENSORS
        #*
        for b in range(0, len(python_list), 1):
            tensor_batch.append( torch.from_numpy(python_list[b]).type(data_type) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        return tensor_batch

    def batch_to_label_vector(self, python_list):
        """
        Convert the input Python list including batches to a numpy column vector for evaluating metrics.
        
        Inputs
        ----------
        - 'python_list': Python list with batches as numpy arrays

        Outputs
        ----------
        - 'tensor_batch': 
        """
        
        #*################
        #* ERROR CHECKER
        #*
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list, list) ):
            raise TypeError("Expected list as input. Received input of type: ", str(type(python_list)) )
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list[0], np.ndarray) ):
            raise TypeError("Expected numpy arrays as input. Received input of type: ", str(type(python_list[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        return np.concatenate(python_list, axis = 0)
    #*
    #*#### END DEFINED METHODS #####
    #*##############################
#*
#*#### DatasetManager class  #####
#*################################


#*#############################
#*#### CubeManager class  #####
#*
# todo: (Optional) Add error checker to warn user when loaded ground-truths have different number of total labels
class CubeManager:
    """
    This class is used to load '_cropped_Pre-processed.mat' files destined to work with Neural Network models, both for training and classification.
    It stores the ground-truth map coordenates of every sample in such a way that indeces among attributes correspond to the same sample.
    Batches have the samples, their corresponding labels as well as their X and Y coordenates.
    - Recomendation: Create 2 instances of CubeManager, 1 to load all data to train and another 1 to load only the image to classify.
                     This way, in this second instance we can know the coordenates of the predicted batches.
    - Note: 'CubeManager' only works with numpy arrays!
    
    """

    #*##########################
    #*#### DEFINED METHODS #####
    #*
    def __init__(self, patch_size = 7, batch_size = 16, dic_label = None, batch_dim = '2D'):
        """
        Define the constructor of 'CubeManager' class. It only works with '_cropped_Pre-processed.mat' files (GT and preProcessedImages).

        Inputs
        ----------
        - 'patch_size': Integer. Size of the 3D patches.
        - 'batch_size': Integer. Size of each data batch.
        - 'dic_label':  Python dictionary. Includes labels as keys and label4Classes as values.
        - 'batch_dim':  String flag to indicate if batches will be '2D' (batches of single pixels of the ground-truth) or '3D' (batches with patches of pixels, taking as
        reference a pixel from the ground-truth and its surroundings.)
            - 2D: Pixels with spectral information. Row of 1 sample or pixel and all columns indicating spectral information.
            - 3D: Small pixel images representing spatial information ('patch_size' of heigh and width) where each pixel indicates its spectral information.

        Attributes
        ----------
        - Global attributes:
            - 'patch_size':         Integer corresponding the size of the 3D patches. Attibute used to calculate the padding to add to the hsi cubes.
            - 'batch_size':         Integer representing the size of each batch.
            - 'dic_label':          Python dictionary with the labels as keys and their corresponding label4Classes as values. Used to generate datasets from cubes.
            - 'batch_dim':          String indicating if batches are '2D' or '3D'.
            - 'pad_margin':         Calculated pad dimension using 'patch_size' to add to each patient cube (usefull to create 3D patches)
            - 'max_gt_width':       Calculated maximum width of all loaded cubes. Used inside '__create_3D_batches()' to create an empty array to append all ground-truth maps and preProcessedImages.
            - 'max_padded_width':   Calculated maximum width of all loaded and padded cubes.  Used inside '__create_3D_batches()' to create an empty array to append all ground-truth maps and preProcessedImages.
        - Attribute related to '_cropped_Pre-processed.mat' files (GT and preProcessedImages):
            - 'patients_list':    Python list. Attribute to store all patient IDs as a python list with strings. Used when _cropped_Pre-processed.mat files are loaded.
            - 'patient_cubes':          Python dictionary. Indeces are the patient IDs. Stores each patient 'preProcessedImage' (as cube) and 'groundTruthMap' (as gt). Dictionary keys are:
                - 'pad_preProcessedImage':  Padded Preprocessed cubes data for every patient. Used to create the patches.
                - 'pad_groundTruthMap':     Padded Ground truth maps for every patient. Used to create the patches.
                - 'raw_preProcessedImage':  Raw Preprocessed cubes data for every patient. Used to predict data.
                - 'raw_groundTruthMap':     Raw Ground truth maps for every patient. Used to predict data.
                - 'label_coords':           Numpy array with the (x, y) coordenates and the label for every labeled pixel in the ground-truth map (if batches are 2D) or the padded ground-truth map (if batches are 3D).
            - 'data':               Numpy array (but initialized as Python list). Attribute to store all dataset pixels once all patient cubes have been loaded. All data will be appended.
            - 'label':              Numpy array (but initialized as Python list). Attribute to store all dataset labels once all patient cubes have been loaded. All labels will be appended.
            - 'label4Classes':      Numpy array (but initialized as Python list): Attribute to store all dataset labels 4 classes once all patient cubes have been loaded. All label4Classes will be appended.
            - 'label_coords':       Numpy array (but initialized as Python list): Attribute to store all coordenates for labeled pixels once all patient cubes have been loaded. All coordenates will be appended.
                                    2D numpy array with 3 columns (x, y, patientNum), to properly identify to which patient the coordenates correspond.
                                    'patientNum' is an integer index that help us identify the patient_id. When loading patients, the order of the IDs in the Python list will comply with this index.
                                    Example: If patient ID35C02 is the first in the python list, then 'patientNum' would be 0. To ensure that, we can execute:
                                        > input:    print(cm_train.patients_list[cm_train.label_coords[0, -1]])
                                        > output:   ID35C02
                                        Where 'cm_train' is a 'CubeManager' instance, 'patients_list' is one of its attributes as well as 'label_coords'. By using 'label_coords[0, -1]' we get the 'patientNum'.
            - 'numUniqueLabels':    Integer. Attribute to store the total number of different labels once all patients have been loaded.
            - 'numTotalSamples':    Integer. Attribute to store the total number of samples once all patients have been loaded. 
            - 'numBands':           Integer. Number of loaded spectral bands
            - 'appended_cubes':     Numpy array. All appended padded preProcessedImages cubes loaded in a single array. Generated when '__append_loaded_cubes()' is called. Used when calling '__create_3D_batches()'.
            - 'appended_gtMaps':    Numpy array. All appended padded ground-truth maps loaded in a single array. Generated when '__append_loaded_cubes()' is called. Used when calling '__create_3D_batches()'.
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if patch_size is an integer number
        if not ( isinstance(patch_size, int) ):
            raise TypeError("Expected integer (int) as input. Received input of type: ", str(type(patch_size)) )
        # Check if batch_size is an integer number
        if not ( isinstance(batch_size, int) ):
            raise TypeError("Expected integer (int) as input. Received input of type: ", str(type(batch_size)) )
        # Check if batch_dim is '2D' or '3D
        if not (batch_dim == '2D' or batch_dim == '3D'):
            raise RuntimeError("To create batches, please specify their dimensions. Use '2D' or '3D as input for 'batch_dim'.")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        #* CREATE ATTRIBUTES FOR THE INSTANCES
        # Global attributes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.dic_label = dic_label
        self.batch_dim = batch_dim

        self.pad_margin = int(np.ceil(self.patch_size/2))
        self.max_gt_width = 0
        self.max_padded_width = 0

        #* Attribute related to '_cropped_Pre-processed.mat' files (GT and preProcessedImages)
        self.patients_list = []
        self.patient_cubes = {}

        self.data = []
        self.label = []
        self.label4Classes = []
        self.label_coords = []

        self.numUniqueLabels = None
        self.numTotalSamples = None
        self.numBands = None

        self.appended_cubes = None
        self.appended_gtMaps = None
    
    def __get_gt_labels_coord(self, gt_map, preProcessedImage, patientNum, batch_dim):
        """
        (Private method) Get the coordenates for each label in the passed ground truth map
        - Important: This method works if a call to or 'load_patient_cubes()' was made first.

        Inputs
        ----------
        - 'gt_map':             Numpy array with the ground truth map.
        - 'preProcessedImage':  Numpy array with the preProcessedImage.
        - 'patientNum':         Integer to indicate the patient id of the input cube.

        Outputs
        ----------
        - 'labels_coord': 2D numpy array with 3 columns:
            - i) x coordenates for every labeled pixel in the ground truth map
			- ii) y coordenates for every labeled pixel in the ground truth map
			- iii) Label for the pixel
        """

        # Create empty array to append all label coordenates with 1 row and 3 colums (x, y, label) 
        labels_coord = np.zeros((1,3))

        # Flag to indicate that we have deleted the first row of the temporary array,
        # otherwise we would append always the empty row.
        deleted_row = False

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PATIENTS IN THE INPUT LIST
        #* EXCEPT THE LABEL 0, WHCIH IS UNLABELED DATA
        #*
        for label in np.unique(gt_map)[1::]:
            # [1::] indicates the following thing:
            #	> '1': start a the second element of the list
            #	> ':': end at the last element
            #	> ':': jump in invervals of 1

            # Extract all the coordenates for the 'label' available in the Ground Truth.
            # np.nonzero() returns a tuple of arrays, one for each dimension of gt_map == label.
            x, y = np.nonzero(gt_map == label)

            #*############################################################
            #* IF STATEMENT TO CHECK IF BATCHES ARE '3D' AND NEED TO USE
            #* THE PADDED GROUND-TRUTH MAP 
            #*
            if (batch_dim == '3D'):
                # If '3D' batches, sum the pad margin to the extracted coordenates. This way, we can ensure
                # that we are using the correct coordenates for the padded ground-truth map
                x = x + self.pad_margin
                y = y + self.pad_margin
            #*
            #* END OF IF
            #*############

            # Call private method to generate dataset from the passed cube
            self.__get_dataset_from_cube(preProcessedImage, x, y, label, patientNum)

            # Create and append a 2D numpy array with 3 columns:
			#	- x: x coordenates for every 'label' pixel
			#	- y: y coordenates for every 'label' pixel
			# 	- np.ones(x.shape[0],) * label]): vector with the same lenght as the number of sampled pixels of label 'l'.
            #                                     Note: labels are passed to label4Classes!
            labels_coord = np.vstack((labels_coord, np.array([x, y, np.ones(x.shape[0],) * label], dtype=int).transpose()))

            #*############################################################
            #* IF STATEMENT TO CHECK IF FIRST ROW OF THE TEMP ARRAY HAS
            #* BEEN DELETED. (remember they where created with the first 
            #* row as empty)
            #*
            if not (deleted_row):
                # Delete the first empty row
                labels_coord = np.delete(labels_coord, 0, axis = 0)
                # Update flag to True
                deleted_row = True
            #*
            #* END OF IF
            #*############
        #*
        #* END FOR LOOP
        #*##############

        return labels_coord
 
    def __get_dataset_from_cube(self, preProcessedImage, x, y, label, patientNum):
        """
        (Private method) Generate a dataset from the input cube (preProcessedImage).
        Append the dataset generated to the instance attributes 'self.data', 'self.cubes_label'
        and 'self_cubes_label4Classes'.
        - Important: This method works if a call to or '__get_gt_labels_coord()' was made first.

        Inputs
        ----------
        - 'preProcessedImage':  Numpy array with the preProcessed image.
        - 'x':                  Numpy array with the X coordenates.
        - 'y':                  Numpy array with the Y coordenates.   
        - 'label':              Label value of all passed X and Y coordenates.
        - 'patientNum':         Integer to indicate the patient id of the input coordenates.    
        """

        # Append to 'self.label_coords' all coordenates from the preProcessedImage
        self.label_coords.append( np.array([x, y, np.ones(x.shape)*patientNum]).transpose() )

        # Append to 'self.data' all pixels from the preProcessedImage corresponding to the input label
        self.data.append( preProcessedImage[x, y] )

        # Create temporary array for label and label4Classes with same rows as the preProcessedImage
        temp_array = np.ones((preProcessedImage[x, y].shape[0], 1))

        # Get the label4Class from the passed label
        label4Class = self.__label_2_label4Class(label)

        # Append to 'self.cubes_label' and 'self.cubes_label4Classes' a numpy array with same
        # rows ad the preProcessedImage but filled with different values.
        self.label.append( temp_array*label )
        self.label4Classes.append( temp_array*label4Class )

    def __label_2_label4Class(self, label):
        """
        (Private method) Extract the label4Class for the corresponding input label.
        Finds in the label dictionary created the value of the corresponding label key.
        (Keys should be string, that is why we cast label with str() )

        Inputs
        ----------
        - 'label': Integer value.          
        """
        return self.dic_label.get(str(label))

    def load_patient_cubes(self, patients_list, dir_path_gt, dir_par_preProcessed):
        """
        Load all patient '.mat' ground truth maps and its corresponding preProcessedImage from the input list 'patients_list'.
        It saves the data in 8 'CubeManager' attributes:
        - 'self.patients_list':     Saves the input python list to have it as reference in the instance.
        - 'self.data':              Append all ground-truth pixels from all loaded patients by appending them.
        - 'self.label':             Append all ground-truth labels from all loaded patients by appending them.
        - 'self.label4Classes':     Append all ground-truth label4Classes from all loaded patients by appending them.
        - 'self.label_coords':      Append all ground-truth coordenates from all loaded patients by appending them and indicate to which patient each sample corresponds to.
                                    (2D Numpy array with 3 columns: x coordenate, y coordenate, patient number as an index to indicate the order of the loaded patients)
        - 'self.numUniqueLabels':   Stores the number of unique classes from all stored labels.
        - 'self.numTotalSamples':   Stores the number of total samples loaded.
        - 'self.patient_cubes': Python dictionary. Indeces are the patient IDs. Stores each patient 'preProcessedImage' (as cube) and 'groundTruthMap' (as gt). Dictionary keys are:
                - 'pad_preProcessedImage':  Padded Preprocessed cubes data for every patient. Used to create the patches.
                - 'pad_groundTruthMap':     Padded Ground truth maps for every patient. Used to create the patches.
                - 'raw_preProcessedImage':  Raw Preprocessed cubes data for every patient. Used to predict data.
                - 'raw_groundTruthMap':     Raw Ground truth maps for every patient. Used to predict data.
                - 'label_coords':           Numpy array with the (x, y) coordenates and the label for every labeled pixel in the ground-truth map.
        
        If more than 1 patient is given in the list, the data is appended to the dictionary, so that each index in the attribute corresponds to 1 single patient.
        - Important: '_cropped_Pre-processed.mat' files need to have 'groundTruthMap' (ground-truth) and 'preProcessedImage' (preprocessed image) name fields.

        Inputs
        ----------
        - 'patients_list':          Python list including the strings ID for each patient.
        - 'dir_path_gt':            String that includes the path directory where the ground truth files are.
        - 'dir_par_preProcessed':   String that includes the path directory where the preProcessed image files are.
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if python list is empty
        if (len(patients_list) == 0):
            raise RuntimeError("Not expected an empty python list input. 'patients_list' is empty.")
        # Check if first python list element is a string
        if not ( isinstance(patients_list[0], str) ):
            raise TypeError("Expected first element of 'patients_list' to be string. Received instead element of type: ", str(type(patients_list[0])) )
        # Once the first element of the list is a string, check if all the elements in the 'patients_list' are also string
        if (len(patients_list) != 1):
            if ( all(element == patients_list[0] for element in patients_list) ):
                raise TypeError("Expected 'patients_list' to only contain string elements. Please ensure all elements in the list are of type 'str' ")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        # Save in the 'self.patients_list' attribute all patient IDs as a python list with strings
        self.patients_list = patients_list

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PATIENTS IN THE INPUT LIST.
        #* IT ALSO APPENDS ALL GT MAPS AND PREPROCESSED IMAGES IN 
        #* 'self.gtMap' AND 'self.preProcessedImage' ATTRIBUTES
        #*
        p = 0   # Variable to indicate with an index the loaded patient

        for patient in patients_list:

            gt_mat = loadmat(dir_path_gt + 'SNAPgt' + patient + '_cropped_Pre-processed.mat')                           # Load ground truth map from the current patient
            preProcessed_mat = loadmat(dir_par_preProcessed + 'SNAPimages' + patient + '_cropped_Pre-processed.mat')    # Load preProcessed image from the current patient
        
            #*####################################################################
            #* IF STATEMENT TO CHECK IF CURRENT GROUND-TRUTH MAP IS WIDER 
            #* THAN THE MAXIMUM GROUND TRUTH WIDTH, AND UPDATE 'self.max_gt_width'
            #*
            if(gt_mat['groundTruthMap'].shape[1] >= self.max_gt_width):
                self.max_gt_width = gt_mat['groundTruthMap'].shape[1]
                self.max_padded_width = self.max_gt_width + 2*self.pad_margin
            #*
            #* END OF IF
            #*############
            
            # Get coordenate labels for the current patient ground truth map.
            # Inside, it calls '__get_dataset_from_cube()' method to append data to the
            # instance attributes 'self.data', 'self.cubes_label' and 'self.cubes_label4Classes'.
            labels_cords = self.__get_gt_labels_coord(gt_mat['groundTruthMap'], preProcessed_mat['preProcessedImage'], patientNum = p, batch_dim = self.batch_dim)

            # Apply a constant padding to the GT to the height and width dimensions
            gt = np.pad(gt_mat['groundTruthMap'], (self.pad_margin,), 'constant')
            # Apply a constant padding to the preProcessed cube to the height and width dimensions (not the spectral channels)
            cube = np.pad(preProcessed_mat['preProcessedImage'], [(self.pad_margin, self.pad_margin), (self.pad_margin, self.pad_margin), (0,0)], 'constant')

            self.patient_cubes[patient] = {'pad_preProcessedImage': cube, 'pad_groundTruthMap': gt,
                                         'raw_preProcessedImage': preProcessed_mat['preProcessedImage'],'raw_groundTruthMap': gt_mat['groundTruthMap'],
                                         'label_coords': labels_cords}
            
            p += 1  # Update patient number variable
        #*
        #* END FOR LOOP
        #*##############

        # Call private instance method to concatenate properly 'self.data', 'self.cubes_label'
        # and 'self.cubes_label4Classes'. At this point they are Python lists where each element
        # corresponds to a specific label and data for every patient loaded. (If 2 images with 5
        # classes were loaded, these attributes would have 10 elements). That is why we call the
        # 'concatenate_list_to_numpy' method, to append all data.
        self.data = self.concatenate_list_to_numpy(self.data)
        self.label = self.concatenate_list_to_numpy(self.label).astype('int')
        self.label4Classes = self.concatenate_list_to_numpy(self.label4Classes).astype('int')
        self.label_coords = self.concatenate_list_to_numpy(self.label_coords).astype('int')

        self.numUniqueLabels = len(np.unique(self.label))       # Store in the 'self.numUniqueLabels' attribute a numpy array with the number of unique classes from all stored labels
        self.numTotalSamples = self.data.shape[0]               # Store in the 'self.numTotalSamples' attribute the total number of loaded samples

        self.__append_loaded_cubes()    # Create numpy arrays with all appended and padded ground-truth maps and preProcessedImages

    def __append_loaded_cubes(self):
        """
        (Private method) Reads all loaded ground-truth maps and preProcessedImages and saves 2 main numpy arrays as instance attributes:
        'self.appended_cubes' and'self.appended_gtMaps'. This method is called at the end of 'load_patient_cubes()' (when all data has been loaded).
        These attributes include the padded ground-truth maps and preProcessedImages loaded in 'load_patient_cubes()' method.
        """

        # Calculate the number of bands length of the first loaded cube
        self.numBands = self.patient_cubes[self.patients_list[0]]['pad_preProcessedImage'].shape[-1]

        # Create first row of temp arrays used to append all ground-truth maps and preProcessedImages
        padded_gt_maps = np.zeros((1, self.max_padded_width))
        padded_preProcessedImages = np.zeros((1, self.max_padded_width, self.numBands))

        # Flag to indicate that we have deleted the first row of the temporary array,
        # otherwise we would append always the empty row.
        deleted_row = False

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PATIENTS IN THE INPUT LIST.
        #*
        for patient in self.patients_list:

            # Load the patient preProcessedImage and ground-truth map
            temp_preProcessedImage = self.patient_cubes[patient]['pad_preProcessedImage']
            temp_gt_map = self.patient_cubes[patient]['pad_groundTruthMap']

            #*############################################################
            #* IF STATEMENT TO CHECK IF CURRENT PATIENT GROUND-TRUTH MAP 
            #* IS SMALLER THAN THE MAXIMUM LOADED GROUND-TRUTH WIDTH
            #*
            if ( temp_gt_map.shape[-1] < self.max_padded_width):
                # Calculate the padding to add (so that we can append all ground-truth maps with same width)
                padding = self.max_padded_width - temp_gt_map.shape[-1]
                # Add padding to the current ground-truth map and preProcessedImage to the right of the array
                temp_preProcessedImage = np.pad(temp_preProcessedImage, [(0, 0), (0, padding), (0,0)], 'constant')
                temp_gt_map = np.pad(temp_gt_map, [(0, 0), (0, padding)] , 'constant')
            #*
            #* END OF IF
            #*############

            # Concatenate the current patient cube and ground-truth map
            padded_preProcessedImages = np.vstack((padded_preProcessedImages, temp_preProcessedImage))
            padded_gt_maps = np.vstack((padded_gt_maps, temp_gt_map))

            #*############################################################
            #* IF STATEMENT TO CHECK IF FIRST ROW OF THE TEMP ARRAY HAS
            #* BEEN DELETED. (remember they where created with the first 
            #* row as empty)
            #*
            if not (deleted_row):
                # Delete the first empty row
                padded_preProcessedImages = np.delete(padded_preProcessedImages, 0, axis = 0)
                padded_gt_maps = np.delete(padded_gt_maps, 0, axis = 0)
                # Update flag to True
                deleted_row = True
            #*
            #* END OF IF
            #*############
        #*
        #* END FOR LOOP
        #*##############

        # Store appended padded ground-truth maps and preProcessedImages in the instance atributes
        self.appended_cubes = padded_preProcessedImages
        self.appended_gtMaps = padded_gt_maps.astype(int)

    def __largest_class(self):
        """
        (Private method) Look for the labeled class with more elements from a numpy vector.
        - If working with '2D' batches, we look at all unique labels of 'self.label4Classes' attribute.
        - If working with '3D' batches, we look at all unique labels of 'self.appended_gtMaps' attribute (except unlabeled pixels representing 0)
        - Important: This method works if a call to 'load_patient_datasets()' or 'load_patient_cubes()' was made first.

        Inputs
        ----------
        - 'data_flag': String indicating if we want to evaluate 'datasets' or 'cubes'.
        
        Outputs
        ----------
        - Integer number indicating the label with largest number of elements
        """
        # Create empty array with the same dimensions as the number of unique labels
        temp_labels = np.zeros(self.numUniqueLabels, dtype=int)

        
        #*#################################################
        #* IF ELSE STATEMENT TO CHECK IF WE ARE CREATING 
        #* '2D' OR '3D' BATCHES
        #*
        if (self.batch_dim == '2D'):

            #*############################################################
            #* FOR LOOP ITERATES OVER ALL AVAILABLE LABELS FROM THE
            #* LOADED OR GENERATED DATASET
            #*
            # Iterate over all available labels
            for label in np.unique(self.label4Classes)[0::]:

                # Number of pixels that are 'label'
                temp_labels[label-1] = np.count_nonzero(self.label4Classes == label)
            
                # Return the label containing the largest amount of elements (+1 since np.where returns the index starting at 0)
                return (np.where(temp_labels == np.amax(temp_labels))[0] + 1)
            #*
            #* END FOR LOOP
            #*##############
        
        elif (self.batch_dim == '3D'):

            #*############################################################
            #* FOR LOOP ITERATES OVER ALL AVAILABLE LABELS FROM THE
            #* LOADED PREPROCESSED CUBES EXCEPT LABEL 0 (which is 
            #* unlabeled data)
            #*
            l = 0
            for label in np.unique(self.appended_gtMaps)[1::]:
                # Number of pixels that are 'label'
                temp_labels[l] = np.count_nonzero(self.appended_gtMaps == label)

                l += 1
            #*
            #* END FOR LOOP
            #*##############

            # Return the index label containing the largest amount of elements 
            label_index = (np.where(temp_labels == np.amax(temp_labels))[0])

            return np.asscalar(np.unique(self.appended_gtMaps)[1::][label_index])   # We need to use np.asscalar() because we want to retrieve the numpy scalar as a Python scalar.
        #*
        #* END OF IF ELSE
        #*################

    def create_batches(self):
        """
        Public method that calls the private methods create_2D_batches() or create_3D_batches() depending on which
        'batch_dim' was given in the instance initialization.

        Outputs
        ----------
        - Python dictionary with 3 Python lists: (they are all in order, so index 0 of any key value would have information of the same sample)
            - A) key = 'data'.          Includes 'list_samples': Python list with sample of batches
            - B) key = 'label4Classes'. Includes 'list_labels': Python list with the labels of all batches in 'list_samples'
            - C) key = 'label_coords'.  Includes 'list_coords': Python list with the label coordenates of all batches elements in 'list_coords'
            - D) key = 'patientNums'.   Includes 'list_patientNum_batches': Python list with the patient identifier of all batches elements created
        """

        if (self.batch_dim == '2D'):
            return self.__create_2D_batches()
        elif (self.batch_dim == '3D'):
            return self.__create_3D_batches()

    def __create_2D_batches(self):
        """
        Create a Python dictionary with small 2D batches of size 'batch_size' from the loaded cubes. It follows the Random Stratified Sampling methodology.
        Not all batches will be perfectly distributed, since classes with fewer samples may not appear in all batches. Also, in case a batch is not going to comply with
        the input 'batch_size', we add more pixels from the class with more samples.
        Note: If we have few samples left but they are not as big as 'batch_size', then we discard those pixels. (batches need to should be the same size for training)
        Important: This method works when '_cropped_Pre-processed.mat' files have been loaded! Therefore, we do store pixel coordenates!

        Outputs
        ----------
        - Python dictionary with 3 Python lists: (they are all in order, so index 0 of any key value would have information of the same sample)
            - A) key = 'data'.          Includes 'list_samples': Python list with sample of batches
            - B) key = 'label4Classes'. Includes 'list_labels': Python list with the labels of all batches in 'list_samples'
            - C) key = 'label_coords'.  Includes 'list_coords': Python list with the label coordenates of all batches elements in 'list_coords'
            - D) key = 'patientNums'.   Includes 'list_patientNum_batches': Python list with the patient identifier of all batches elements created
        """
        #*################
        #* ERROR CHECKER
        #*
        # Check if 'patient_cubes' instance attribute contains elements. If not, it would mean that no '_cropped_Pre-processed.mat' file has been loaded.
        if ( len(self.patient_cubes) == 0):
            raise RuntimeError("No '_cropped_Pre-processed.mat' file has been loaded. To use 'create_2d_batches()' method, please first load datasets using the 'load_patient_cubes()' method.")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        largest_label = self.__largest_class()                      # Find the label with more elements

        data_temp = np.copy(self.data)                              # Create a copy of the loaded data in a temporary variable. This way, we don't delete data from the instance 'data' attribute.
        label4Classes_temp = np.copy(self.label4Classes)            # Create a copy of the loaded label4Classes in a temporary variable. This way, we don't delete data from the instance 'label4Classes' attribute.
        label_coords_temp = np.copy(self.label_coords)              # Create a copy of the loaded label_coords in a temporary variable. This way, we don't delete data from the instance 'label_coords' attribute.

        list_sample_batches = []
        list_label_batches = []
        list_coords_batches = []
        list_patientNum_batches = []

        #*###############################################
        #* WHILE LOOP CREATES 1 BATCH EVERY ITERATION
        #*
        while data_temp.shape[0] >= self.batch_size:                # Stay in this while loop if the number of samples left in 'dataTemp' is equal or greater than 'bath_size'

            list_samples = []                                       # Create empty Python list to append data samples
            list_labels = []                                        # Create empty Python list to append data labels
            list_coords = []                                        # Create empty Python list to append label coordenates
            list_patientNums = []                                   # Create empty Python list to append the patient number corresponding to each label

            size_current_batch = 0

            num_total_samples_left =  data_temp.shape[0]
            
            #*#################################################
            #* FOR LOOP ITERATES OVER EVERY LABEL AVAILABLE
            #*
            for label in np.unique(label4Classes_temp)[0::]:

                # [0::] indicates the following thing:
                #	- '0': start a the first element of the list
                #	- ':': end at the last element
                #	- ':': jump in invervals of 1
                class_indices = np.where(label4Classes_temp == label)[0]        # Extract the 'indices' where pixels are labeled as the input 'label'. [0] is to extract the indices, since np.where() returns 2 arrays.
                                                                                # 'class_indices' is an array where each position indicates the index of every labeled pixel equal to 'label'
                
                #*###############################################################################
                #* IF STATEMENT TO EVALUATE IF THERE ARE SAMPLES LEFT FOR THE CURRENT LABEL
                #*
                if ( len(class_indices) > 0 ):

                    percentage =  (len(class_indices) / num_total_samples_left)     # Calculate percentage that correspond to the pixel 'labels' out of all samples left unused

                    num_samples = int(round(self.batch_size * percentage))          # Calculate the number of samples to add to the batch for the current label

                    if num_samples == 0: num_samples = 1
                    
                    #*##############################################################
                    #* IF STATEMENT TO EVALUATE WHETHER OR NOT I NEED TO
                    #* SUBSTRACT ANY SAMPLE BECAUSE THE BATCH HAS REACH ITS LIMIT
                    #*
                    if ( (size_current_batch + num_samples) > self.batch_size ):

                        samples_to_substract = ((size_current_batch + num_samples) - self.batch_size)   # Calculate the samples that need to be substracted
                        num_samples -= samples_to_substract                                             # Update the 'num_samples' variable to comply with the 'batch_dimension' size
                        class_indices = class_indices[0:-samples_to_substract]                          # Delete the latest samples that we are going to substract
                    #*
                    #* END OF IF
                    #*############

                    size_current_batch += num_samples                                                           # Update 'size_current_batch' variable to know the size of the current batch

                    sample_indices = np.random.choice(len(class_indices), num_samples, replace=False)           # Randomly select a total of 'num_samples' sample indices for the batch

                    list_samples.append(data_temp[class_indices[sample_indices]])                               # Store in the Python list all randomly selected samples from the current label
                    list_labels.append(label4Classes_temp[class_indices[sample_indices]])                       # Store in the Python list all randomly selected sample labels from the current label
                    list_coords.append(label_coords_temp[class_indices[sample_indices]][:, 0:-1])               # Store in the Python list all randomly selected sample label coordenates from the current label
                    list_patientNums.append(label_coords_temp[class_indices[sample_indices]][:, -1])            # Store in the Python list the patient number of the randomly selected samples for the current label

                    data_temp = np.delete(data_temp, class_indices[sample_indices], axis = 0)                       # Remove the sampled pixels from the temporary 'data_temp' variable
                    label4Classes_temp = np.delete(label4Classes_temp, class_indices[sample_indices], axis = 0)     # Remove the sampled labels from the temporary 'label4Classes_temp' variable
                    label_coords_temp = np.delete(label_coords_temp, class_indices[sample_indices], axis = 0)       # Remove the sampled coordenates from the temporary 'label_coords' variable
                #*
                #* END OF IF
                #*############
            #*
            #* END OF FOR LOOP
            #*###################

            # Convert both Python lists 'list_samples' and 'list_labels' to numpy arrays by using 'np.concatenate()'.
            # This way we concatenate all pixels from all labels to be stored in 1 single variable, which would represent 1 single batch
            single_sample_batch = np.concatenate(list_samples, axis=0)
            single_label_batch = np.concatenate(list_labels, axis=0)
            single_coords_batch = np.concatenate(list_coords, axis=0)
            single_patientNums_batch = np.concatenate(list_patientNums, axis=0)
            # 'single_patientNums_batch' have shape (N,), but should be (N,1). That is why we do the reshape. Otherwise we would encounter errors when
            # entering the next if statement (since when using np.vstack() at least 1 array should have (N,1) shape.)
            single_patientNums_batch = single_patientNums_batch.reshape((len(single_patientNums_batch), 1))

            #*##################################################################
            #* IF STATEMENT TO ADD ADDITIONAL SAMPLES TO THE CURRENT BATCH 
            #* IN CASE ITS LENGHT IS LESS THAN THE ACTUAL BATCH_SIZE
            #* (We take samples from the label with more classes)
            #*
            if (len(single_sample_batch) < self.batch_size):
                # Calculate the number of elements that we need to add to the batch. Will use the 'label' with more elements
                samples_to_add = (self.batch_size - len(single_sample_batch))

                # Extract the 'indices' where pixels are labeled as the 'largest label'. [0] is to extract the indices, since np.where() returns 2 arrays.
                class_indices_temp = np.where(label4Classes_temp == largest_label)[0]

                # Randomly select a total of 'num_samples' sample indices for the batch
                sample_indices_temp = np.random.choice(len(class_indices_temp), samples_to_add, replace=False)

                # Store in the Python list all randomly selected samples and labels from the current label and the additional samples from the 'largest label' 
                single_sample_batch = np.vstack([single_sample_batch, data_temp[class_indices_temp[sample_indices_temp]]])
                single_label_batch = np.vstack([single_label_batch, label4Classes_temp[class_indices_temp[sample_indices_temp]]])
                single_coords_batch = np.vstack([single_coords_batch, label_coords_temp[class_indices_temp[sample_indices_temp]][:, 0:-1]])
                single_patientNums_batch = np.vstack([single_patientNums_batch, label_coords_temp[class_indices_temp[sample_indices_temp]][:, -1]])

                # Remove the additional sampled pixels and labels from the temporary 'data','label4Classes' and 'label_coords_temp' variables
                data_temp = np.delete(data_temp, class_indices_temp[sample_indices_temp], axis = 0)
                label4Classes_temp = np.delete(label4Classes_temp, class_indices_temp[sample_indices_temp], axis = 0)
                label_coords_temp = np.delete(label_coords_temp, class_indices_temp[sample_indices_temp], axis = 0)
            #*   
            #* END OF IF
            #*##############

            list_sample_batches.append(single_sample_batch)
            list_label_batches.append(single_label_batch)
            list_coords_batches.append(single_coords_batch)
            list_patientNum_batches.append(single_patientNums_batch)
        #*
        #* END OF WHILE 
        #*################
        
        """
        #*########################################################
        #* IF STATEMENT IS USED TO APPEND THE REMAINING DATA 
        #* THAT CAN NOT BE USED AS A BATCH OF 'batch_size' SIZE
        if( (data_temp.shape[0] / self.batch_size) > 0):
            list_sample_batches.append(data_temp)
            list_label_batches.append(label4Classes_temp)
            list_coords_batches.append(label_coords_temp[:, 0:-1])

            # Since 'label_coords_temp[:, -1]' extracts a 1D numpy array, we have to reshape it into a 2D numpy array
            # This way we append the array correctly.
            label_coords_temp = label_coords_temp[:, -1].reshape((len(label_coords_temp[:, -1]), 1))
            list_patientNum_batches.append(label_coords_temp)

        #*   
        #* END OF IF
        #*##############
        """

        return {'data':list_sample_batches, 'label4Classes':list_label_batches, 'label_coords': list_coords_batches, 'patientNums': list_patientNum_batches}

    def __create_3D_batches(self):
        """
        Create a Python dictionary with batches composed of small patches images (3D batches). It access the
        appended ground-truth maps and appended cubes attributes to generate 3D patches. These attributes
        were created at the end of 'load_patient_cubes()', which calls '__append_loaded_cubes()'.
        Note: If we have few samples left but they are not as big as 'batch_size', then we discard those pixels. (batches need to should be the same size for training)
        Important: This method works when '_cropped_Pre-processed.mat' files have been loaded!

        Inputs
        ----------
        - 'python_dic': Python dictionary obtained after calling the '__create_2D_batches()' method.

        Outputs
        ----------
        - Python dictionary with 3 Python lists: (they are all in order, so index 0 of any key value would have information of the same sample)
            - A) key = 'cube'.          Includes 'list_cube_batch': Python list with sample of batches
            - B) key = 'label'.         Includes 'list_labels_batch': Python list with the labels of all batches in 'list_cube_batch'. Each batch has a numpy array with
                                        - X: x coordenate of the center pixel of every batch (the ground-truth map labeled pixel)
                                        - Y: y coordenate of the center pixel of every batch (the ground-truth map labeled pixel)
                                        - Label: indicating the label of the patch (takes the ground-truth label of the center pixel)
        """
        #*################
        #* ERROR CHECKER
        #*
        # Check if 'patient_cubes' instance attribute contains elements. If not, it would mean that no '_cropped_Pre-processed.mat' file has been loaded.
        if ( len(self.patient_cubes) == 0):
            raise RuntimeError("No '_cropped_Pre-processed.mat' file has been loaded. To use 'create_2d_batches()' method, please first load datasets using the 'load_patient_cubes()' method.")
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        largest_label = self.__largest_class()                  # Find the label with more elements

        gt_maps = np.copy(self.appended_gtMaps)                 # Create a copy of the appended and padded ground-truth maps in a temporary variable. This way, we don't delete data from the instance 'appended_gtMaps' attribute.            
        
        # Create empty Python lists
        list_labels_batch = []
        list_cube_batch = []

        #*###############################################
        #* WHILE LOOP CREATES 1 BATCH EVERY ITERATION
        #*
        while np.sum(gt_maps > 0) >= self.batch_size:                # Stay in this while loop if the number of samples left in 'dataTemp' is equal or greater than 'bath_size'

            list_cube_samples = []                                       # Create empty Python list to append data cube samples
            list_label_samples = []                                      # Create empty Python list to append data ground truth map label samples

            size_current_batch = 0

            num_total_samples_left =  gt_maps[gt_maps > 0].shape[0]

            #*#################################################
            #* FOR LOOP ITERATES OVER EVERY LABEL AVAILABLE
            #*
            for label in np.unique(gt_maps)[1::]:

                # [1::] indicates the following thing:
                #	- '1': start a the second element of the list (omit 0 labels)
                #	- ':': end at the last element
                #	- ':': jump in invervals of 1

                x, y = np.nonzero(gt_maps == label)     # Extract the x, y coordenates where pixels are labeled as the input 'label'.

                #*###############################################################################
                #* IF STATEMENT TO EVALUATE IF THERE ARE SAMPLES LEFT FOR THE CURRENT LABEL
                #*
                if ( len(x) > 0 ):

                    percentage =  (len(x) / num_total_samples_left)             # Calculate percentage that correspond to the pixel 'labels' out of all samples left unused

                    num_samples = int(round(self.batch_size * percentage))      # Calculate the number of samples to add to the batch for the current label

                    if num_samples == 0: num_samples = 1

                    #*##############################################################
                    #* IF STATEMENT TO EVALUATE WHETHER OR NOT I NEED TO
                    #* SUBSTRACT ANY SAMPLE BECAUSE THE BATCH HAS REACH ITS LIMIT
                    #*
                    if ( (size_current_batch + num_samples) > self.batch_size ):

                        samples_to_substract = ((size_current_batch + num_samples) - self.batch_size)   # Calculate the samples that need to be substracted
                        num_samples -= samples_to_substract                                             # Update the 'num_samples' variable to comply with the 'batch_dimension' size
                        x = x[0:-samples_to_substract]                                                  # Delete the latest samples that we are going to substract
                        y = y[0:-samples_to_substract]                                                  # Delete the latest samples that we are going to substract
                    #*
                    #* END OF IF
                    #*############
                    
                    size_current_batch += num_samples                                                       # Update 'size_current_batch' variable to know the size of the current batch

                    sample_indices = np.random.choice(len(x), num_samples, replace=False)                   # Randomly select a total of 'num_samples' sample indices for the batch

                    # Append a numpy array to the created Python list 'list_label_samples'.
                    # We create a 2D numpy array with 3 columns including the sampled coordenates destined to the training sample and the label of the pixel.
                    #	> x[sample_indices] --> x coordenate of label 'label' pixel
                    #	> y[sample_indices] --> y coordenate of label 'label' pixel 
                    # 	> np.ones(x[sample_indices].shape[0],)*label --> creates a vector with the same lenght as the number of sampled pixels of label 'l'.
                    #										  This vector does not contain 1s but the actual value of the label!
                    label4Class = self.__label_2_label4Class(label)     # Convert 'label' to a 'label4Class' to properly create batches and feed CNN with labels starting at 1
                    list_label_samples.append(np.array([x[sample_indices], y[sample_indices], np.ones(x[sample_indices].shape[0],)*label4Class], dtype=int).transpose())
                    list_cube_samples.append(self.__get_patches(x[sample_indices], y[sample_indices]))

                    # Delete from the ground truth the pixel coordenates that have been used to create the patches
                    gt_maps[x[sample_indices], y[sample_indices]] = 0
                    

                #*
                #* END OF IF
                #*############
            #*
            #* END OF FOR LOOP
            #*###################

            # Convert both Python lists 'list_samples' and 'list_labels' to numpy arrays by using 'np.concatenate()'.
            # This way we concatenate all pixels from all labels to be stored in 1 single variable, which would represent 1 single batch
            single_label_batch = np.concatenate(list_label_samples, axis = 0)
            single_cube_batch = np.concatenate(list_cube_samples, axis = 0)

            #*##################################################################
            #* IF STATEMENT TO ADD ADDITIONAL SAMPLES TO THE CURRENT BATCH 
            #* IN CASE ITS LENGHT IS LESS THAN THE ACTUAL BATCH_SIZE
            #* (We take samples from the label with more classes)
            #*
            if (len(single_label_batch) < self.batch_size):
                # Calculate the number of elements that we need to add to the batch. Will use the 'label' with more elements
                samples_to_add = (self.batch_size - len(single_label_batch))

                # Extract all left coordenates where pixels are labeled as the 'largest label'.
                x_temp, y_temp = np.where(gt_maps == largest_label)

                # Randomly select a total of 'num_samples' sample indices for the batch
                sample_indices_temp = np.random.choice(len(x_temp), samples_to_add, replace=False)

                # Store in the Python list all randomly selected samples and labels from the current label and the additional samples from the 'largest label' 
                label4Class = self.__label_2_label4Class(largest_label)     # Convert 'largest_label' to a 'label4Class' to properly create batches and feed CNN with labels starting at 1
                single_label_batch = np.vstack([single_label_batch, np.array([x_temp[sample_indices_temp], y_temp[sample_indices_temp], np.ones(x_temp[sample_indices_temp].shape[0],)*label4Class], dtype=int).transpose()])
                single_cube_batch = np.vstack([single_cube_batch, self.__get_patches(x_temp[sample_indices_temp], y_temp[sample_indices_temp])])

                # Delete from the ground truth the pixel coordenates that have been used to create the patches
                gt_maps[x_temp[sample_indices_temp], y_temp[sample_indices_temp]] = 0
            #*   
            #* END OF IF
            #*##############

            list_labels_batch.append(single_label_batch)
            list_cube_batch.append(single_cube_batch)
        
        #*
        #* END OF WHILE 
        #*################

        """
        #*########################################################
        #* IF STATEMENT IS USED TO APPEND THE REMAINING DATA 
        #* THAT CAN NOT BE USED AS A BATCH OF 'batch_size' SIZE
        if( (np.sum(gt_maps > 0) / self.batch_size) > 0):

            # Create temporary array for the left over samples with 3 colums, (X, Y, label)
            temp_label_array = []
            temp_sample_array = []

            #*###################################################
            #* FOR LOOP ITERATES OVER EVERY LABEL LEFT AVAILABLE
            #*
            for label in np.unique(gt_maps)[1::]:
                # Extract the remaining coordenates
                x_left, y_left = np.where(gt_maps == label)

                # Stack left over samples coordenates and label to the temporary array
                temp_label_array.append( np.array([x_left, y_left, np.ones(x_left.shape[0],) * label]).transpose() )
                temp_sample_array.append( self.__get_patches(x_left, y_left) )
            
            #*
            #* END OF FOR LOOP
            #*###################

            # Concatenate all labels from the remaining labels and get a numpy array
            # Append the remaining ground truth labels to 'list_gt_maps_batch'
            # Generate patches with the remaining coordenates and append to batch list
            list_labels_batch.append( self.concatenate_list_to_numpy(temp_label_array) ) 
            list_cube_batch.append( self.concatenate_list_to_numpy(temp_sample_array) )  

        #*   
        #* END OF IF
        #*##############
        """

        return {'cube': list_cube_batch, 'label': list_labels_batch}

    def create_cube_batch(self):
        """
        """
        #*################
        #* ERROR CHECKER
        #*
        if not (len(self.patients_list) == 1):
            raise RuntimeError("'patients_list' contains more than 1 ID. Please, create a new instance of CubeManager and load only 1 image to batch its HSI cube.")
        #*
        #* ERROR CHECKER
        #*################

        # Create a copy of the entire HSI padded cube from the loaded patient image
        cube = np.copy(self.patient_cubes[self.patients_list[0]]['pad_preProcessedImage'])
        gt = np.copy(self.patient_cubes[self.patients_list[0]]['raw_groundTruthMap'])
        
        # Create mask of 1s with same dimension as the loaded ground truth map
        mask = np.ones( (gt.shape[0], gt.shape[1]) )

        # Create empty Python lists to append all batches with 3D patches
        list_cube_batch = []
        list_coords_batch = []

        #*###############################################
        #* WHILE LOOP CREATES 1 BATCH EVERY ITERATION
        #*
        while np.sum(mask > 0) >= self.batch_size:                # Stay in this while loop if the number of samples left in 'dataTemp' is equal or greater than 'bath_size'
            
            list_cube_samples = []                                       # Create empty Python list to append data cube samples
            list_coord_samples = []                                      # Create empty Python list to append data ground truth map label samples

            # Extract the coordenates from all pixels left
            x, y = np.nonzero(mask == 1)

            # From all pixels left, extract 'self.batch_size' random coordenates without replacement
            indices = np.random.choice(len(x), self.batch_size, replace=False)

            # Create a numpy array with the coordenates for the sampled indices and append to the list.
            list_coord_samples.append( np.array([x[indices], y[indices]]).transpose() )

            # Generate 3D patches from the extract coordenates using the patient HSI cube
            # Since the mask does not have the padding, we have to add the pad_margin when accessing the cube
            # since the cube has padding.
            list_cube_samples.append( self.__get_patches_full_cube(x[indices] + self.pad_margin, y[indices] + self.pad_margin, cube) )

            # Delete from the mask the used coordenates to create the patches
            mask[x[indices],y[indices]] = 0

            # Convert both Python lists 'list_cube_samples' and 'list_coord_samples' to numpy arrays by using 'np.concatenate()'.
            # This way we concatenate all pixels from all labels to be stored in 1 single variable, which would represent 1 single batch
            cube_batch = np.concatenate(list_cube_samples, axis=0)
            coords_batch = np.concatenate(list_coord_samples, axis=0)

            # Append each batch to its corresponding list
            list_cube_batch.append(cube_batch)
            list_coords_batch.append(coords_batch)
        #*
        #* END OF WHILE 
        #*################

        #*########################################################
        #* IF STATEMENT IS USED TO APPEND THE REMAINING DATA 
        #* THAT CAN NOT BE USED AS A BATCH OF 'batch_size' SIZE
        #*
        if ( mask[mask > 0].shape[0] / self.batch_size > 0):

            # Extract the coordenates for the remaining left pixels in the mask
            x, y = np.nonzero(mask != 0)
            
            # Create a numpy array with the coordenates for the sampled indices and append to the list
            list_coords_batch.append(np.array([x, y]).transpose())

            # Generate 3D patches from the extract coordenates using the patient HSI cube.
            # Since the mask does not have the padding, we have to add the pad_margin when accessing the cube
            # since the cube has padding.
            list_cube_batch.append(self.__get_patches_full_cube(x + self.pad_margin, y + self.pad_margin, cube))

        #*   
        #* END OF IF
        #*##############

        return {'data': list_cube_batch, 'coords': list_coords_batch}

    def __get_patches_full_cube(self, x, y, cube):
        """
        (Private method) Create 3D patches using the input coordenates and extract data from the entire
        preprocessed cube passed as input. 
        
        Inputs
        ----------
        - 'x' and 'y':  Coordenates to access the input 'cube' and use them as center coordenates
                        for the patches.
        - 'cube':       Numpy array. Preprocessed cube with already added padding.

        Outputs
        ----------
        - 'patches':    Numpy array with all generated patches from the centered coordenates passed as inputs.
        """
    
        # Create empty 'len(x)' arrays (or patches) with size: "number of bands" x "patch_size" x "patch_size" 
        patches = np.zeros((len(x), cube.shape[-1], self.patch_size, self.patch_size))

		# Extract start coordenates for 'x' and 'y' Python lists passed as parameter 
        xs = (x - int(self.patch_size/2)).astype(int)
        ys = (y - int(self.patch_size/2)).astype(int)

		# Extract end coordenates for 'x' and 'y' Python lists passed as parameter
        xe = xs + self.patch_size
        ye = ys + self.patch_size

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PASSED PIXEL COORDENATES TO 
        #* GENERATE PATCHES FROM THE HSI CUBES AND SAVE THEM IN
        #* THE 'patches' VARIABLE.
        #*
        for i in range(len(x)):
            # Save inside the 'patches' variable each small 3D patch of size "number of bands" x "patch_size" x "patch_size"
		    # Use 'self.cube' attribute which contains the 'preProcessed' image. From the 'preProcessed' image extract small patches
		    # from the coordenates extracted.
            patches[i,:,:,:] = np.transpose(cube[xs[i]:xe[i], ys[i]:ye[i], :], (2, 0, 1) )

        #*
        #* END FOR LOOP
        #*##############

        return patches


    def __get_patches(self, x, y):
        """
        (Private method) Uses the input coordenates to extract patches from the appended ground-truth maps
        and appended cubes. Patches are created with dimension (batch_size, num_features, height, width)
        to comply with PyTorch convolutional layer requirements.
        
        Inputs
        ----------
        - 'x' and 'y':  Coordenates to access the 'appended_gtMaps' attribute and use them as center coordenates
                        for the patches.

        Outputs
        ----------
        - 'patches':    Numpy array with all generated patches from the centered coordenates passed as inputs.
        """
    
        # Create empty 'len(x)' arrays (or patches) with size: "number of bands" x "patch_size" x "patch_size" 
        patches = np.zeros((len(x), self.appended_cubes.shape[-1], self.patch_size, self.patch_size))

		# Extract start coordenates for 'x' and 'y' Python lists passed as parameter 
        xs = (x - int(self.patch_size/2)).astype(int)
        ys = (y - int(self.patch_size/2)).astype(int)

		# Extract end coordenates for 'x' and 'y' Python lists passed as parameter
        xe = xs + self.patch_size
        ye = ys + self.patch_size

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL PASSED PIXEL COORDENATES TO 
        #* GENERATE PATCHES FROM THE HSI CUBES AND SAVE THEM IN
        #* THE 'patches' VARIABLE.
        #*
        for i in range(len(x)):
            # Save inside the 'patches' variable each small 3D patch of size "number of bands" x "patch_size" x "patch_size"
		    # Use 'self.cube' attribute which contains the 'preProcessed' image. From the 'preProcessed' image extract small patches
		    # from the coordenates extracted.
            patches[i,:,:,:] = np.transpose(self.appended_cubes[xs[i]:xe[i], ys[i]:ye[i], :], (2, 0, 1) )

        #*
        #* END FOR LOOP
        #*##############

        return patches	

    def concatenate_list_to_numpy(self, python_list):
        """
        Concatenate all elements in the input Python list to return a numpy array.

        Inputs
        ----------
        - 'python_list':    Python list to concatenate.
            
        Outputs
        ----------
        - Numpy array with all elements of the python list concatenated
        """

        #*##############################################################
        #* IF ELSE STATEMENT TO CHECK SHAPE OF THE ELEMENTS IN THE INPUT
        #* PYTHON LIST. GENERATE DIFFERENT TEMPORARY ARRAYS DEPENDING
        #* ON THE INPUT BATCHES (2d with shape 2 and 3d with shape 4)
        #*
        if ( len(python_list[0].shape) == 2 ):
            # If entered here, we are working with data with only rows and columns (samples x wavelenghts)
            # Some examples are: 'self.data', 'self.label4Classes', 'self.label' or 'self.labels_coords'
            # Create temporary array of 1 empty row with same columns as the elements in the python list
            temp_array = np.zeros((1, python_list[0].shape[-1]))

        elif ( len(python_list[0].shape) == 4 ):
            # If entered here, we are working with data with ("patch_id" x "number of bands" x "patch_size" x "patch_size")
            # Create temporary array of 1 array of ("patch_size" x "patch_size" x "number of bands")
            temp_array = np.zeros_like(python_list[0])[0, :, :, :].reshape(1, python_list[0].shape[1], python_list[0].shape[2], python_list[0].shape[3])
        #*
        #* END OF IF ELSE
        #*################

        # Flag to indicate that we have deleted the first row of the temporary array,
        # otherwise we would append always the empty row.
        deleted_row = False

        #*############################################################
        #* FOR LOOP ITERATES OVER ALL ELEMENTS IN THE INPUT LIST
        #* AND STACK THEM IN THE TEMPORARY ARRAY
        #*
        for element in python_list:
            temp_array = np.vstack((temp_array, element))

            #*############################################################
            #* IF STATEMENT TO CHECK IF FIRST ROW OF THE TEMP ARRAY HAS
            #* BEEN DELETED. (remember they where created with the first 
            #* row as empty)
            #*
            if not (deleted_row):
                # Delete the first empty row
                temp_array = np.delete(temp_array, 0, axis = 0)
                # Update flag to True
                deleted_row = True       
            #*
            #* END OF IF
            #*############
        #*
        #* END FOR LOOP
        #*##############

        # Return the numpy array with all elements and delete the first empty row
        return temp_array

    def batch_to_tensor(self, python_list, data_type):
        """
        Convert all numpy array batches included in a Python list to desired PyTorch tensors types.
        
        Inputs
        ----------
        - 'python_list':    Python list with batches as numpy arrays
        - 'data_type':      PyTorch tensor type to convert the numpy array batch to desired tensor type

        Outputs
        ----------
        - 'tensor_batch':   Python list with batches as PyTorch tensors
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list, list) ):
            raise TypeError("Expected list as input. Received input of type: ", str(type(python_list)) )
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list[0], np.ndarray) ):
            raise TypeError("Expected numpy arrays as input. Received input of type: ", str(type(python_list[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        tensor_batch = []                               # Create empty Python list to return

        #*###############################################################
        #* FOR LOOP TO ITERATE OVER ALL BATCHES INCLUDED IN THE INPUT 
        #* PARAMETER 'python_list' AND CONVERT THEM AS PYTORCH TENSORS
        #*
        for b in range(0, len(python_list), 1):
            tensor_batch.append( torch.from_numpy(python_list[b]).type(data_type) )
        #*    
        #* END FOR LOOP
        #*###############

        return tensor_batch

    def batch_to_label_vector(self, python_list):
        """
        Convert the input Python list including batches to a numpy column vector for evaluating metrics.
        
        Inputs
        ----------
        - 'python_list': Python list with batches as numpy arrays

        Outputs
        ----------
        - 'tensor_batch': 
        """
        
        #*################
        #* ERROR CHECKER
        #*
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list, list) ):
            raise TypeError("Expected list as input. Received input of type: ", str(type(python_list)) )
        # Check if the input batches are numpy arrays
        if not ( isinstance(python_list[0], np.ndarray) ):
            raise TypeError("Expected numpy arrays as input. Received input of type: ", str(type(python_list[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        return np.concatenate(python_list, axis = 0)
  
    #*
    #*#### END DEFINED METHODS #####
    #*##############################
#*
#*#### CubeManager class  #####
#*#############################

#*#######################
#* CrossValidator class
#*
class CrossValidator:
    """
    This class uses K-fold cross validator to split the data and to perform double-cross validation
    over PyTorch CNN models.
    
    """
    def __init__(self, batch_data, batch_labels, k_folds=5, numUniqueLabels=None, numBands=25, epochs=100, lr=0.01):
        """
        Define the constructor of 'CrossValidator' class.

        Inputs
        ----------
        - 'batch_data':         Python list. Batches with data. Elements can be numpy arrays or PyTorch tensors.
        - 'batch_labels':       Python list. Batches with labels. Elements can be numpy arrays or PyTorch tensors.
        - 'k_folds':            Integer. Indicates the number of folds for the K-fold cross-validator.
        - 'numUniqueLabels':    Integer. Indicates the number of unique labels in the batches.
        - 'numBands':           Integer. Indicates the spectral bands included in the batches.
        - 'epochs':             Integer. Indicates the number of epochs used to train the CNN models.
        - 'lr':                 Integer. Learning rate used for the optimizer when training CNN models.

        Attributes
        ----------
        - 'calibration_data_folds':     Python list. Each index element includes the numpy data batches destined to train the models for every single Kn-fold split.
        - 'calibration_label_folds':    Python list. Each index element includes the numpy label batches destined to train the models for every single Kn-fold split.
        - 'validation_data_folds':      Python list. Each index element includes the numpy data batches destined to validate the best models for every single Kn-fold split.
        - 'validation_label_folds':     Python list. Each index element includes the numpy label batches destined to validate the best models for every single Kn-fold split.
        - 'test_data_folds':            Python list. Each index element includes the numpy data batches destined to test the best models for every single K-fold split.
        - 'test_label_folds':           Python list. Each index element includes the numpy label batches destined to test the best models for every single K-fold split.
        - 'bestModel':                  Toch. PyTorch model obtained after performing a K-fold double-cross validation.
        """

        self.batch_data = batch_data
        self.batch_labels = batch_labels
        self.k_folds = k_folds
        self.numUniqueLabels = numUniqueLabels
        self.numBands = numBands
        self.epochs = epochs
        self.lr = lr

        self.test_data_folds = None
        self.test_label_folds = None
        self.calibration_data_folds = None
        self.calibration_label_folds = None
        self.validation_data_folds = None
        self.validation_label_folds = None

        self.bestModel = None


    # todo: (Optional) modify the method to work with 2D batches (AT THE MOMENT ONLY WORK WITH 3D BATCHES)
    def __kfold_double_cv_split(self):
        """
        (Private method) Uses the K-fold cross-validator from sklearn to extracts test, calibration, and validation indexes from the input Python list.
        This method generates 3 Python lists with the batches for every K-fold and Kn-fold. Therefore, it returns the splitted data to use
        in a double-cross validation, where each index of 'test_data_folds' is for every K iteration and each indexes of 'calibration_data_folds'
        and 'validation_data_folds' are for every Kn iteration. (The same happens with '_label_folds' lists)
        - Important: At the moment the '__kfold_double_cv_split()' can only be performed if the input Python list 'batches' contains
        elements in 3D (if it includes patches).
        """
        # Create numpy array with same lenght as the number of batches included in the Python list
        # This way we can extract the indices for the batches properly.
        arr_1_loop = np.ones((len(self.batch_data), 1))

        # Create empty lists to store the calibration, validation, and test batches for every K and Kn fold split.
        test_data_folds = []
        test_label_folds = []
        calibration_data_folds = []
        calibration_label_folds = []
        validation_data_folds = []
        validation_label_folds = []

        # Convert Python lists to numpy arrays
        batch_data_array = self.__reshape_list_to_numpy(self.batch_data)
        batch_labels_array = self.__reshape_list_to_numpy(self.batch_labels)

        # General KFold cross-validator using the passed number of 'k_folds'
        # Shuffle = False: consecutive folds will be the shifted version of previous fold.
        kf = KFold(n_splits = self.k_folds, shuffle = False)

        #*#########################################################
        #* FOR LOOP TO EXTRACT ALL INDEXES TO SPLIT THE BATCHES 
        #* FOR TRAINING AND TEST. THIS INDEXES INDICATE WHICH 
        #* BATCHES FROM THE INPUT BATCH LIST THAT ARE DESTINED TO
        #* TRAIN AND TEST.
        #* ----> FIRST CROSS-VALIDATION LOOP (K)
        #*
        for train_k, test_k in kf.split( range(len(arr_1_loop)) ):
            # Each iteration corresponds to 1 single K-fold split.
            # Append the batches using the indexes of the current fold split to the
            # created Python lists.
            test_data_folds.append( batch_data_array[test_k, :, :, :, :]  )
            test_label_folds.append( batch_labels_array[test_k, :, :]  )

            #*#########################################################
            #* FOR LOOP TO EXTRACT ALL INDEXES TO SPLIT THE BATCHES 
            #* FOR CALIBRATION AND VALIDATION. THIS INDEXES INDICATE WHICH 
            #* BATCHES FROM THE INPUT BATCH LIST THAT ARE DESTINED TO
            #* CALIBRATE AND VALIDATE.
            #* ----> SECOND CROSS-VALIDATION LOOP (Kn)
            #*

            for train_kn, test_kn in kf.split( range(len(train_k)) ):
                # Each iteration corresponds to 1 single K-n fold split.
                # Append the batches using the indexes of the current fold split to the
                # numpy batch arrays created in the first loop of the double-cross validation.
                # Note: 'batch_data_array[train_k, :, :, :, :]' is a numpy array with all batches for the Kn folds
                #       Adding '[train_kn, :, :, :, :]' to the end, extracts the batches for calibration while
                #       adding '[test_kn, :, :, :, :]' to the end, extracts the batches for validation.
                calibration_data_folds.append( batch_data_array[train_k, :, :, :, :][train_kn, :, :, :, :] )
                calibration_label_folds.append( batch_labels_array[train_k, :, :][train_kn, :, :] )

                validation_data_folds.append( batch_data_array[train_k, :, :, :, :][test_kn, :, :, :, :] )
                validation_label_folds.append( batch_labels_array[train_k, :, :][test_kn, :, :] )
                
            #*
            #* END FOR LOOP
            #*###############
            
        #*
        #* END FOR LOOP
        #*###############

        # Save all data and label folds into the instance attributes
        self.test_data_folds = test_data_folds
        self.test_label_folds = test_label_folds
        self.calibration_data_folds = calibration_data_folds
        self.calibration_label_folds = calibration_label_folds
        self.validation_data_folds = validation_data_folds
        self.validation_label_folds = validation_label_folds 

    def __reshape_list_to_numpy(self, python_list):
        """
        (Private method) Create a numpy array with +1D to append all elements included in the input 'python_list'.
        It works with batches with data (4D) and labels (2D).
        Used in 'single_cross_validation()' method to properly extract which batches should be used for
        training and which for testing on each K-fold.

        Input
        --------
        - 'python_list': Python list. If working with 4D, each element should have shape '( num_patches, num_features, patch_height, patch_width ).
        If working with 2D, each element should have shape '( num_batch, (x, y, label) )'

        Output
        --------
        It only retrieves 1 output depending of the shape of the input Python List:
        - 'batch_array_5d': Numpy array. In case Python list has 4D. Final shape is '(num_batches, num_patches, num_features, patch_heigh, patch_width)'
        - 'batch_array_3d': Numpy array. In case Python list has 2D. Final shape is '(num_batches, (x, y, label))'
        """

        #*###############################################
        #* IF ELSE STATEMENT TO EVALUATE THE SHAPE OF
        #* THE INPUT PYTHON LIST
        #*
        if (len(python_list[0].shape) == 4):
            # Create empty array with shape '( num_batches, num_patches, num_features, patch_height, patch_width )'
            # The idea is to work with a numpy array where dimensions correspond to '( num_batches, num_patches, num_features, height, width )'
            batch_array_5d = np.zeros((len(python_list), python_list[0].shape[0], python_list[0].shape[1], python_list[0].shape[2], python_list[0].shape[3]))

            #*############################################################
            #* FOR LOOP ITERATES OVER ALL ELEMENTS IN THE INPUT LIST
            #* AND STACK THEM IN THE TEMPORARY ARRAY
            #*
            i = 0
            for element in python_list:

                # Save inside the 'patches' variable each small 3D patch of size "number of bands" x "patch_size" x "patch_size"
                # Use 'self.cube' attribute which contains the 'preProcessed' image. From the 'preProcessed' image extract small patches
                # from the coordenates extracted.
                batch_array_5d[i, :, :, :, :] = element
                i += 1
            
            #*
            #* END FOR LOOP
            #*##############

            return batch_array_5d
        
        elif (len(python_list[0].shape) == 2):

            # Create empty array with shape '(num_batches, x_coord, y_coord, label)'
            # The idea is to work with a numpy array where dimensions correspond to '(num_batches, (x, y, label))'
            batch_array_3d = np.zeros((len(python_list), python_list[0].shape[0], python_list[0].shape[1]))

            #*############################################################
            #* FOR LOOP ITERATES OVER ALL ELEMENTS IN THE INPUT LIST
            #* AND STACK THEM IN THE TEMPORARY ARRAY
            #*
            i = 0
            for element in python_list:

                # Save inside the 'patches' variable each small 3D patch of size "number of bands" x "patch_size" x "patch_size"
                # Use 'self.cube' attribute which contains the 'preProcessed' image. From the 'preProcessed' image extract small patches
                # from the coordenates extracted.
                batch_array_3d[i, :, :] = element
                i += 1
            
            #*
            #* END FOR LOOP
            #*##############
            
            return batch_array_3d
        #*
        #* END OF IF ELSE
        #*###############################################

    def double_cross_validation(self):
        """
        Perform a K-fold double-cross validation and stores in the instance attribute 'self.bestModel' the
        best model. It calls '__kfold_double_cv_split()' internally to split the data for every K and Kn folds.
        Trained models are 'Conv2DNet'.
        """

        print("\tSplitting data before performing K-fold double-cross validation...")
        self.__kfold_double_cv_split()
        print("\tData has been splitted. Performing ", self.k_folds ,"fold double-cross validation...")

        print('\n\t### DOUBLE-CROSS VALIDATION ###')
        #*###############################################################
        #* FOR ITERATION FOR THE OUTER DOUBLE-CROSS VALIDATION LOOP (K)
        #*
        best_K_OACC = 0
        Kn = 0

        for K in range(0, self.k_folds, 1):
            print('\n\t\t Current K fold =', K+1)

            #*################################################################
            #* FOR ITERATION FOR THE INNER DOUBLE-CROSS VALIDATION LOOP (Kn)
            #*
            best_Kn_OACC = 0
            
            for _ in range(0, self.k_folds, 1):
                print('\t\t\t Current Kn fold =', Kn+1)
                
                # Create a Conv2DNet model. We need to define a new one for every Kn iteration
                model = models.Conv2DNet(num_classes = self.numUniqueLabels, in_channels = self.numBands)

                # Convert calibration data to tensor
                batch_x = torch.from_numpy(self.calibration_data_folds[Kn]).type(torch.float)
                batch_y = torch.from_numpy(self.calibration_label_folds[Kn]).type(torch.LongTensor)

                # Train CNN in current Kn fold using the calibration data
                model.trainNet(batch_x = batch_x, batch_y = batch_y, epochs = self.epochs, plot = False, lr = self.lr)

                # Convert validation data to tensor
                batch_x_val = torch.from_numpy(self.validation_data_folds[Kn]).type(torch.float)

                # Test CNN in current Kn fold using the validation data
                y_hat_Kn = model.predict(batch_x = batch_x_val)

                # Manipulate 'validation_label_folds' for the current 'Kn'.
                # We first need to concatenate all batches together with 'np.concatenate()' along the rows (axis=0)
                # Then we extract the labels and not the coordenates (remember that '_labels_folds' variables have (x_coord, y_coord, label)).
                # Since the result is of shape (N,) and the shape of 'y_hat_Kn' is (N, 1), we need to reshape (-1 indicates to take the entire lenght)
                # We convert it as integers since they originally are floats and we need the labels as indexes inside 'get_metrics()'
                y_true_Kn = np.concatenate(self.validation_label_folds[Kn], axis = 0)[:, -1].reshape((-1,1)).astype(int)

                # Evaluate metrics by comparing the predicted labels with the true labels for the current Kn fold
                Kn_OACC = mts.get_metrics(y_true_Kn, y_hat_Kn, self.numUniqueLabels)['OACC']

                if (best_Kn_OACC < Kn_OACC):
                    print('\t\t\t Found new best model!')
                    best_Kn_OACC = Kn_OACC

                    # Save Kn CNN model in local variable
                    best_Kn_model = model
                
                Kn += 1
            #*
            #* END OF INNER DOUBLE-CROSS VALIDATION LOOP (Kn)
            #*################################################

            # Convert test data to tensor
            batch_x_test = torch.from_numpy(self.test_data_folds[K]).type(torch.float)

            # Test 'best_Kn_model' with current K test batch
            y_hat_K = best_Kn_model.predict(batch_x = batch_x_test)

            # Manipulate 'validation_label_folds' for the current 'Kn'.
            # We first need to concatenate all batches together with 'np.concatenate()' along the rows (axis=0)
            # Then we extract the labels and not the coordenates (remember that '_labels_folds' variables have (x_coord, y_coord, label)).
            # Since the result is of shape (N,) and the shape of 'y_hat_Kn' is (N, 1), we need to reshape (-1 indicates to take the entire lenght)
            # We convert it as integers since they originally are floats and we need the labels as indexes inside 'get_metrics()'
            y_true_K = np.concatenate(self.test_label_folds[K], axis = 0)[:, -1].reshape((-1,1)).astype(int)

            # Evaluate metrics by comparing the predicted labels with the true labels for the current K fold
            # Use the last column of labels since is the one containing the labels (others has coordenates)
            K_OACC = mts.get_metrics(y_true_K, y_hat_K, self.numUniqueLabels)['OACC']

            if (best_K_OACC < K_OACC):
                print('\t\t Found new best model!')
                best_K_OACC = K_OACC

                # Save or update best CNN model obtained during double cross-validation
                self.bestModel = best_Kn_model
                
        #*
        #* END OF OUTER DOUBLE-CROSS VALIDATION LOOP (Kn)
        #*#################################################

        print('\n\t### DOUBLE-CROSS VALIDATION IS FINISHED ###')
#*
#* CrossValidator class
#*#######################




# todo: Define method to load raw images ('.tif') // load_patient_rawImages()
# ? Maybe create a new class?