#################################################################################
# Alberto Martín Pérez - 2021
# This concept has been extracted from Guillermo Vázquez Valle dataset_manager.py
#--------------------------------------------------------------------------------
# This script is used to manipulate HSI data and create small patches or batches.
# It uses the groundtruth pixel references as a mask and extract all HS pixels 
# from around them to create patches. It can also use already made .mat datasets.
#################################################################################

import numpy as np                  # Import numpy
import torch                        # Import PyTorch
from scipy.io import loadmat        # Import scipy.io to load .mat files

#*################################
#*#### DatasetManager class  #####
#*
class DatasetManager:
    """
    This class is used to load data destined to work with Neural Network models, both for training and classification.
    - Note: 'DatasetManager' only works with numpy arrays!
    
    """
    #*##########################
    #*#### DEFINED METHODS #####
    #*

    def __init__(self):
        """
        Define the constructor of 'DatasetManager' class.

        Attributes
        ----------
        - 'patients_list': Python list. Attribute to store all patient IDs as a python list with strings
        - 'data': Numpy array (but initialized as Python list). Attribute to store all dataset pixels once all patients have been loaded. All data will be appended
        - 'label': Numpy array (but initialized as Python list). Attribute to store all dataset labels once all patients have been loaded. All labels will be appended
        - 'label4Classes': Numpy array (but initialized as Python list): Attribute to store all dataset labels 4 classes once all patients have been loaded. All label4Classes will be appended
        - 'numUniqueLabels': Integer. Attribute to store the total number of different labels once all patients have been loaded
        - 'numTotalSamples': Integer. Attribute to store the total number of samples once all patients have been loaded
        """

        #* CREATE ATTRIBUTES FOR THE INSTANCES
        # Global attributes
        self.patients_list = []

        #* Attributes related to '_dataset.mat' files
        self.data = []
        self.label = []
        self.label4Classes = []

        self.numUniqueLabels = None
        self.numTotalSamples = None

    def __largest_class(self):
        """
        (Private method) Look for the labeled class with more elements from a numpy vector
        
        Outputs
        ----------
        - Integer number indicating the label with largest number of elements
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if 'self.data' and 'self.label4Classes' attributes are empty
        if ( (len(self.data) == 0) or (len(self.label4Classes) == 0) ):
            raise RuntimeError("Data has not been loaded. Please execute first load_patient_datasets() method." )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        # Create empty array with the same dimensions as the number of unique labels
        temp_labels = np.zeros(self.numUniqueLabels, dtype=int)

        for label in np.unique(self.label4Classes)[0::]:                         # Iterate over all available labels

            temp_labels[label-1] = np.count_nonzero(self.label4Classes == label)        # Number of pixels that are 'label'

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
        #*
        #* END FOR LOOP
        #*##############

        # Store in 'self.data', 'self.label' and 'self.label4Classes' instance attributes all loaded data
        # Get rid of the first empty elements of the temporary arrays (remember they where created with the first row as empty)
        self.data = np.delete(temp_data_array, 0, axis = 0)
        self.label = np.delete(temp_label_array, 0, axis = 0).astype(int)
        self.label4Classes = np.delete(temp_label4Classes_array, 0, axis = 0).astype(int)

        self.numUniqueLabels = len(np.unique(self.label))       # Store in the 'self.numUniqueLabels' attribute a numpy array with the number of unique classes from all stored labels
        self.numTotalSamples = self.data.shape[0]               # Store in the 'self.numTotalSamples' attribute the total number of loaded samples

    # todo: Define method to load both preProcessedImages and ground truth maps // load_patient_cubes()
    # todo: Define method to create 3d batches from the cubes loaded using load_patient_cubes() // create_3d_batches()

    # todo: Define method to load raw images ('.tif') // load_patient_rawImages()

    # todo: Define method to modify labels from one to another

    # todo: Define method to batch a single image

    def create_2d_batches(self, batch_size):
        """
        Create a Python dictionary with small batches of size 'batch_size' from the loaded data and their labels. It follows the Random Stratified Sampling methodology.
        Not all batches will be perfectly distributed, since classes with fewer samples may not appear in all batches. Also, in case a batch is not going to comply with
        the input 'batch_size', we add more pixels from the class with more samples. The last batch would be the only one with fewer samples than 'batch_size'.
        
        Inputs
        ----------
        - 'batch_size': Integer. Represents the size of each batch.
        
        Outputs
        ----------
        - Python dictionary with 2 Python lists:
            - A) key = 'data'. Includes 'list_samples': Python list with sample of batches
            - B) key = 'label4Classes'. Includes 'list_labels': Python list with the labels of all batches in 'list_samples'
        """

        #*################
        #* ERROR CHECKER
        #*
        # Check if batch_size is an integer number
        if not ( isinstance(batch_size, int) ):
            raise TypeError("Expected integer (int) as input. Received input of type: ", str(type(batch_size)) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        largest_label = self.__largest_class()                        # Find the label with more elements

        data_temp = np.copy(self.data)                              # Create a copy of the loaded data in a temporary variable. This way, we don't delete data from the instance 'data' attribute.
        label4Classes_temp = np.copy(self.label4Classes)            # Create a copy of the loaded label4Classes in a temporary variable. This way, we don't delete data from the instance 'label4Classes' attribute.

        list_sample_batches = []
        list_label_batches = []

        #*###############################################
        #* WHILE LOOP CREATES 1 BATCH EVERY ITERATION
        #*
        while data_temp.shape[0] >= batch_size:                     # Stay in this while loop if the number of samples left in 'dataTemp' is equal or greater than 'bath_size'

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

                    num_samples = int(round(batch_size * percentage))               # Calculate the number of samples to add to the batch for the current label

                    if num_samples == 0: num_samples = 1
                    
                    #*##############################################################
                    #* IF STATEMENT IS TO EVALUATE WHETHER OR NOT I NEED TO
                    #*  SUBSTRACT ANY SAMPLE BECAUSE THE BATCH HAS REACH ITS LIMIT
                    #*
                    if ( (size_current_batch + num_samples) > batch_size ):
                        samples_to_substract = ((size_current_batch + num_samples) - batch_size)    # Calculate the samples that need to be substracted
                        num_samples -= samples_to_substract                                         # Update the 'num_samples' variable to comply with the 'batch_dimension' size
                        class_indices = class_indices[0:-samples_to_substract]                      # Delete the latest samples that we are going to substract
                    #*
                    #* END OF IF
                    #*############

                    size_current_batch += num_samples                                                               # Update 'size_current_batch' variable to know the size of the current batch

                    sample_indices = np.random.choice(len(class_indices), num_samples, replace=False)               # Randomly select a total of 'num_samples' sample indices for the batch

                    list_samples.append(data_temp[class_indices[sample_indices]])                                    # Store in the Python list all randomly selected samples from the current label
                    list_labels.append(label4Classes_temp[class_indices[sample_indices]])                            # Store in the Python list all randomly selected sample labels from the current label

                    data_temp = np.delete(data_temp, class_indices[sample_indices], axis = 0)                        # Remove the sampled pixels from the original 'data' variable
                    label4Classes_temp = np.delete(label4Classes_temp, class_indices[sample_indices], axis = 0)      # Remove the sampled labels from the original 'data' variable   
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
            if (len(single_sample_batch) < batch_size):
                # Calculate the number of elements that we need to add to the batch. Will use the 'label' with more elements
                samples_to_add = (batch_size - len(single_sample_batch))

                # Extract the 'indices' where pixels are labeled as the 'largest label'. [0] is to extract the indices, since np.where() returns 2 arrays.
                class_indices_temp = np.where(label4Classes_temp == largest_label)[0]

                # Randomly select a total of 'num_samples' sample indices for the batch
                sample_indices_temp = np.random.choice(len(class_indices_temp), samples_to_add, replace=False)

                # Store in the Python list all randomly selected samples and labels from the current label and the additional samples from the 'largest label' 
                single_sample_batch = np.vstack([single_sample_batch, data_temp[class_indices_temp[sample_indices_temp]]])
                single_label_batch = np.vstack([single_label_batch, label4Classes_temp[class_indices_temp[sample_indices_temp]]])

                # Remove the additional sampled pixels and labels from the original 'data' variable
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
        if( (data_temp.shape[0] / batch_size) > 0):
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