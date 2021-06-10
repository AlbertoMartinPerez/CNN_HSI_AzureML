###############################################################
# Alberto Martín Pérez - 2021
# This script has been developed by Guillermo Vázquez Valle. 
# Some modifications have been done.
#--------------------------------------------------------------
# This script is used to extract metrics from a prediction.
# It is also used to generate classification maps.
###############################################################

import numpy as np					# Import numpy
import matplotlib.pyplot as plt		# Import matplotlib pyplot

#*##########################
#*#### DEFINED METHODS #####
#*
def get_metrics(true_labels, pred_labels, num_clases):
	"""
    Takes true and predicted labels to generate a confusion matrix and extract overall accuracy,
    accuracy, sensitivity, specificity and precission metrics for every input class.

    Inputs
    ----------
    - 'true_labels':	Numpy array with original true labels
    - 'pred_labels':	Numpy array with predicted labels
    - 'num_clases':		Integer with the number of unique classes

	Outputs
    ----------
	- Python dictionary with the following key and values:
		- 'OACC': Overall accuracy value
		- 'SEN': Sensivity python list with each class sensitivity value
		- 'SPE': Specificity python list with each class specificity value
		- 'ACC': Accuracy python list with each class accuracy value
		- 'PRECISION': Precission python list with each class precission value
		- 'CON_MAT': Confusion matrix numpy array
    """

    #*################
    #* ERROR CHECKER
    #*
    # Check if 'true_labels' and 'pred_labels' are numpy arrays
	if not isinstance(true_labels, np.ndarray):
		raise TypeError("Expected numpy array as input. Received instead variable 'true_labels' of type: ", str(type(true_labels)) )
	if not isinstance(pred_labels, np.ndarray):
		raise TypeError("Expected numpy array as input. Received instead variable 'pred_labels' of type: ", str(type(pred_labels)) )

	# Check if 'true_labels' and 'pred_labels' have the same shape
	if not (true_labels.shape == pred_labels.shape):
		raise RuntimeError("Expected 'labels' and 'pred_labels' to have the same shape. Received true_labels.shape = ", str(true_labels.shape), " and pred_labels.shape = ", str(pred_labels.shape))

	# Check if 'true_labels' and 'pred_labels' are numpy column vectors
	if not (true_labels.shape[1] == 1 and pred_labels.shape[1] == 1):
		raise RuntimeError("Expected 'true_labels' and 'pred_labels' to be column vectors of shape (N, 1). Received true_labels.shape = ", str(true_labels.shape), " and pred_labels.shape = ", str(pred_labels.shape))
    #*    
    #* END OF ERROR CHECKER ###
    #*#########################

	# Create empty confusion matrix with dimensions 'num_clases'
	confusion_mx = np.zeros([num_clases, num_clases], dtype='int')

    #*##################################################
    #* FOR LOOP ITERATES OVER ALL INPUT LABELS
    #* ON EACH ITERATION WE ADD 1 TO THE CORRESPONDING
	#* CONFUSION MATRIX INDEX.
    #*
	for i in range(true_labels.shape[0]):
    	# We substract 1 to the index since numpy true_labels[i] or pred_labels[i] contain integers > 0
		# Remember that confusion_mx is a numpy array and index starts at 0.
		confusion_mx[true_labels[i]-1, pred_labels[i]-1] += 1
	#*
    #* END FOR LOOP
    #*##############

	# Call private method '__class_metrics()' to obtain sensivity, specifity, accuracy and precission 
	# vectors where each element corresponds to the metric obtained in each class.
	sensivity, specifity, accuracy, precission = __class_metrics(confusion_mx)

	# Compute the overall accuracy
	oac = np.sum(np.diag(confusion_mx)) / np.sum(confusion_mx)

	return {'OACC':oac, 'SEN':sensivity, 'SPE':specifity, 'ACC':accuracy, 'PRECISION':precission, 'CON_MAT': confusion_mx}

def __class_metrics(confusion_mx):
	"""
    (Private method) Computes the sensitivity, specificity, accuracy and precission metrics
    for every class available in the input confusion matrix.

    Inputs
    ----------
    - 'confusion_mx': Confusion matrix generated in 'get_metrics()' method.

	Outputs
    ----------
	- 'sensivity':	Sensivity python list with each class sensitivity value
	- 'specifity':	Specificity python list with each class specificity value
	- 'accuracy':	Accuracy python list with each class accuracy value
	- 'precission':	Precission python list with each class precission value
    """

	# Create empty Python list for evert metric
	sensivity = [] 
	specifity = []
	accuracy = []
	precission = []

	epsilon = 10e-8
	
	#*####################################################
    #* FOR LOOP ITERATES OVER THE INPUT CONFUSION MATRIX
    #* ON EACH ITERATION WE COMPUTE THE TP, TN, FP and FN
	#* FOR THE CORRESPONDING LABEL. THEN WE COMPUTE THE
	#* SEN, SPE, ACC and PRE METRICS FOR THAT LABEL.
    #*
	for i in range(confusion_mx.shape[0]):

		tp = confusion_mx[i,i]
		fn = np.sum(confusion_mx[i,:])-tp
		fp = np.sum(confusion_mx[:,i])-tp
		tn = np.sum(confusion_mx)-tp-fp-fn

		sensivity.append(tp/(tp+fn+epsilon))
		specifity.append(tn/(tn+fp+epsilon))
		accuracy.append((tn+tp)/(tn+tp+fn+fp+epsilon))
		precission.append(tp/(tp+fp+epsilon))
	#*
    #* END FOR LOOP
    #*##############

	return sensivity, specifity, accuracy, precission


# TODO: Review this function before using it
	# todo: I) Implement first 'load_patient_groundTruthMap()' method inside hsi_dataManager.py
def get_prediction_map(true_labels, pred_labels, dims, title="", plot = True, save = False):
	"""
	Generate a subplot with the original ground-truth and the predicted ground-truth.

    Inputs
    ----------
    - 'true_labels':	Numpy array with original true labels
    - 'pred_labels':	Numpy array with predicted labels
    - 'dims':			Python list containing the dimensions of the classified ground truth map
    - 'title':			String to set the title of the subplot
    - 'plot':			Boolean flag to indicate whether or not to plot the subplot
    - 'save':			Boolean flag to indicate whether or not to save the subplot
    - 'save_path':		String variable containing the path to save the subplot

    """
	#*################
    #* ERROR CHECKER
    #*
    # Check if 'true_labels' and 'pred_labels' are numpy arrays
	if not isinstance(true_labels, np.ndarray):
		raise TypeError("Expected numpy array as input. Received instead variable 'true_labels' of type: ", str(type(true_labels)) )
	if not isinstance(pred_labels, np.ndarray):
		raise TypeError("Expected numpy array as input. Received instead variable 'pred_labels' of type: ", str(type(pred_labels)) )

	# Check if 'true_labels' and 'pred_labels' have the same shape
	if not (true_labels.shape == pred_labels.shape):
		raise RuntimeError("Expected 'labels' and 'pred_labels' to have the same shape. Received true_labels.shape = ", str(true_labels.shape), " and pred_labels.shape = ", str(pred_labels.shape))

	# Check if 'true_labels' and 'pred_labels' are numpy column vectors
	if not (true_labels.shape[1] == 1 and pred_labels.shape[1] == 1):
		raise RuntimeError("Expected 'true_labels' and 'pred_labels' to be column vectors of shape (N, 1). Received true_labels.shape = ", str(true_labels.shape), " and pred_labels.shape = ", str(pred_labels.shape))
    #*    
    #* END OF ERROR CHECKER ###
    #*#########################

	gt_raw_map = np.zeros((dims[0], dims[1]))
	pred_raw_map = np.zeros((dims[0], dims[1]))

	gt_raw_map[true_labels[:,0], true_labels[:,1]] = true_labels[:,2]
	pred_raw_map[true_labels[:,0], true_labels[:,1]] = pred_labels

	gt_color = convert2color(gt_raw_map)
	preds_color =  convert2color(pred_raw_map)


	# Plot the results
	if plot:
		fig = plt.figure()
		#fig.suptitle(title, fontsize=16)
		fig.add_subplot(1, 2, 1)
		plt.imshow(preds_color)
		plt.title("Prediction")
		plt.axis('off')
		fig.add_subplot(1, 2, 2)
		plt.imshow(gt_color)
		plt.title("Ground truth")
		plt.axis('off')
		plt.show()

	# Save the model
	if save:
		plt.savefig(title+'_map.png', bbox_inches='tight')

# TODO: Review this function before using it
def get_cube_prediction_map(preds, coords, dims, title="", plot=True, save=False):

	preds_raw_map = np.zeros((dims[0], dims[1]))

	#targets_raw_map[targets[:,0],targets[:,1]] = targets[:,2]
	preds_raw_map[coords[:,0],coords[:,1]] = preds

	#targets_color = convert2color(targets_raw_map)
	preds_color =  convert2color(preds_raw_map)


	# Plot the results
	fig = plt.figure()
	#fig.suptitle(title, fontsize=16)
	plt.imshow(preds_color)
	plt.axis('off')


	if plot:
		plt.show()

	if save:
		plt.savefig(title+'_cube_map.png', bbox_inches='tight')

# TODO: Review this function before using it
def paletteGen():

	palette = {0: (0, 0, 0)}

	palette[0] = np.asarray(np.array([0,0,0])*255,dtype='uint8')
	palette[1] = np.asarray(np.array([0,1,0])*255,dtype='uint8')
	palette[2] = np.asarray(np.array([1,0,0])*255,dtype='uint8')
	palette[3] = np.asarray(np.array([0,0,1])*255,dtype='uint8')
	palette[4] = np.asarray(np.array([0,.63,.89])*255,dtype='uint8')
	#palette[5] = np.asarray(np.array([.49,0,1])*255,dtype='uint8')
	palette[5] = np.asarray(np.array([1,0,1])*255,dtype='uint8')
	palette[6] = np.asarray(np.array([1,1,1])*255,dtype='uint8')

	return palette

# TODO: Review this function before using it
def convert2color(gt_raw, palette=paletteGen()):
	

	# zeros MxNx3
	gt_color = np.zeros((gt_raw.shape[0], gt_raw.shape[1], 3), dtype=np.uint8)

	for c, i in palette.items():
		# get mask of vals that gt == #item in palette
		m = gt_raw == c
		# set color val in 3 components
		gt_color[m] = i

	return gt_color

#*
#*#### END DEFINED METHODS #####
#*##############################