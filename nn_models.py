#################################################################################
# Alberto Martín Pérez - 2021
#--------------------------------------------------------------------------------
# This script is used to create a PyTorch Neural Network to train and classify
#################################################################################

import torch                        # Import Pytorch
import torch.nn as nn               # Import Pytorch nn module
import torch.nn.functional as F     # Import Pytorch nn.functional as F
from tqdm import tqdm               # Import tqdm, a python library used to add progress bars that show the processing behind the execution of the program
import matplotlib.pyplot as plt     # Import matplotlib to create loss and accuracy plot
import numpy as np                  # Import numpy


# ? GPU FUNCTIONALITY HERE
# Save in 'device' whether we use the CPU or the GPU via CUDA to train Neural Networks
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

#*###############################
#*#### FourLayerNet class  #####
#*
class FourLayerNet(nn.Module):
    """
    This class define all the layers of the FourLayerNet network and packs them by
    enheriting from the 'Module' class defined by Pytorch.
    - Important: Using 'self' within this class means using the model itself (self = self.model)
    - Important: This network works with 2D data!

    Layers
    ----------
    - self.linear1 = torch.nn.Linear(D_in, H)
    - self.linear2 = torch.nn.Linear(H, H*2)
    - self.linear3 = torch.nn.Linear(H*2, H*3)
    - self.linear4 = torch.nn.Linear(H*3, D_out)
    """

    #*#######################################
    #*#### DEFINED FourLayerNet METHODS #####
    #*
    def __init__(self, D_in, H, D_out):
        """
        Constructor to define the Neural Network model elements. All classes have a function __init__(), which is always executed when the class
	    is bein initiated. It is used to assign vlaues to objet properties or other operations
	    that are necessary to do when the object is being created. 
        
        Inputs
        ----------
        - D_in:     Input dimensions for the first layer
        - H:        Dimension for the hidden layers
        - D_out:    Output dimension for the latest layer
        """

        super(FourLayerNet, self).__init__()

        #* DEFINE NETWORK LAYERS
        # For 'Linear' layers, input shape should be the number of input features in the first layer
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H*2)
        self.linear3 = torch.nn.Linear(H*2, H*3)
        self.linear4 = torch.nn.Linear(H*3, D_out)

    def forward(self, x):
        """
        Encapsulate the forward computational steps. PyTorch call the nested model to perform the forward pass.
        DO NOT call the forward(x) method. You should call the whole model itself, as in model(x) to perform a forward pass and output predictions.
        (Within this class, a simple call to self(x) will do the work). It receives a PyTorch tensor as input to compute the pass and then return and output tensor.
        
        Inputs
        ----------
        - x:    Input PyTorch tensor used to compute the forward pass
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        return x

    def trainNet(self, batch_x, batch_y, epochs = 500, plot = False, lr = 0.002):
        """
        Train the FourLayerNet Neural Network.
        
        Inputs
        ----------
        - batch_x:  Python list containing PyTorch tensor batches destined for training
        - batch_y:  Python list containing PyTorch tensor batches labels destined for training
        - epochs:   Number of epochs to run over the training data
        - plot:     Flag to whether or not plot the loss and accuracy per epoch
        - lr:       Learning rate used in the optimizer
        """
    	# Define two empty arrays that will store, for each epoch, the cost and the accuracy
    	# These arrays basically are as big as the number of epochs (or iterations) over the
    	# the training set, meaning that each position on these arrays will have the cost and
    	# the accuracy for every epoch.
    	# They will be used to plot graphs and determine if the training is good or not.
        loss_train = np.zeros(epochs)
        accuracy = np.zeros(epochs)

        # Loss and Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)    # 'self' is the model itself. We are basically doing 'model.parameters()'
        criterion = torch.nn.CrossEntropyLoss()

        # ? GPU FUNCTIONALITY HERE
        # Store the model inside the GPU memory
        # self.cuda()

        # Set the model to train mode to let know PyTorch that during backpropagation
        # it should not apply drop-out, batch norm or any layer with special behaviours
        # that behabe differently on the train and test procedures. 
        self.train()        # 'self' is the model itself. We are basically doing 'model.train()'

        #*#######################################################################
        #* FOR LOOP TO TRAIN THE MODEL WITHT THE CORRESPONDING NUMBER OF EPOCHS
        #* IT ALSO SHOWS A PROGRESS BAR THAT INCREASES ON EVERY EPOCH
        #*
        print("\nStarted training your Neural Network of type: ", str(type(self)))

        for epoch in tqdm(range(epochs)):

            running_loss = 0.0
            correct_train = 0.0

            #*###############################################################
            #* FOR LOOP TO ITERATE OVER ALL TRAINING BATCHES ON EVERY EPOCH
            #*
            for X, Y in zip(batch_x, batch_y):

                # ? GPU FUNCTIONALITY HERE
                # Transfer the current batch tensors to the GPU if available
                # X = X.to(device)
                # Y = Y.to(device)

                # Forward pass. This will automatically call the 'forward(self, x)' method
                y_pred = self(X)    # 'self' is the model itself. We are basically doing 'model(X)'

                # ? GPU FUNCTIONALITY HERE
                # Transfer the created tensor to the GPU
                # y_pred = y_pred.to(device)

                # Compute loss.
                # Loss function needs the predicted outputs from 'model()' and a row vector with the same number of elements as the number of entries (rows) in 'y_pred'
                # The problem is that loading 'label4Classes' and convert them as Pytorch tensor, we need to manipulate it, since the shape of the number of labels stored
                # in 'y' has shape of 'torch.Size([10, 1])' when it should be 'torch.Size([10])'. To achieve that, we can transpose the vector to have 1 row and N columns (using '.T') and
                # then extract the first row ('[0]').
                # Please note: 'torch.nn.CrossEntropyLoss()' needs labels starting from 0 to Number of classes -1. That is why we apply -1 since labels should start at 0 and not 1, as saved in 'label4Classes'
                loss = criterion(y_pred, Y.T[0]-1 )

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # Apply softmax and get the value of the tensor with highest probability. Store the predicted class in 'predicted
                predicted =  torch.argmax(F.softmax(y_pred, dim = 1), dim = 1)
                # Sum the correct train of the current mini-batch
                correct_train += ((predicted == Y.T[0]-1).sum().item()) / (predicted.shape[0])
                # Sum the loss of the current mini-batch
                running_loss += loss.item()

                loss_train[epoch] = running_loss/len(batch_x)
                accuracy[epoch] = correct_train/len(batch_x)
            #*
            #* END FOR LOOP
            #*##############
        
        print("Finished training! Your model is now ready to predict.\n")
        #*
        #* END FOR LOOP
        #*##############

        # Plot training loss error
        if(plot):
            plt.title('Train loss and accuracy')
            plt.xlabel('epoch')
            plt.plot(loss_train, 'r-', label = 'loss')
            plt.plot(accuracy, 'g-', label = 'accuracy')
            plt.legend()
            plt.show()

    def predict(self, batch_x):
        """
        Predict 2D batches of data with a FourLayerNet model
        
        Inputs
        ----------
        - 'batch_x':        Python list containing PyTorch tensor batches destined for training
        
        Outputs
        ----------
        - 'pred_labels':    Python list containing numpy arrays with the labels for every element in every batch
        """ 
        #*################
        #* ERROR CHECKER
        #*
        # Check if input parameter is a python list
        if not ( isinstance(batch_x, list) ):
            raise RuntimeError("Expected a python list as input. Received instead element of type: ", str(type(batch_x)) )
        # Check if python list is empty
        if (len(batch_x) == 0):
            raise RuntimeError("Not expected an empty python list input. 'batch_x' is empty.")
        # Check if first python list element is a PyTorch tensor
        if not ( isinstance(batch_x[0], torch.Tensor) ):
            raise TypeError("Expected first element of 'batch_x' to be 'torch.Tensor'. Received instead element of type: ", str(type(batch_x[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        # Create empty Python lists to store all labels predicted for every batch
        pred_labels = []

        with torch.no_grad():
            self.eval()             # 'self' is the model itself. We are basically doing 'model.eval()' 

            #*##################################################
            #* FOR LOOP TO ITERATE OVER ALL BATCHES
            #* Used to predict the labels for the input batches
            #*
            for X in batch_x:

                # ? GPU FUNCTIONALITY HERE
                # Transfer the current batch tensor to the GPU if available
                # X = X.to(device)
                
                # For every single batch 'X', we calculate the label probabilities for every element.
                # 'ps' is an array where each row represents the probabilities of each element to be one of the output classes returned by the model.
                # If the batch contains N elements, then 'ps' is an array with N rows and M columns, where M indicates the number of outputs that returns the model.
                ps = F.softmax(self(X), dim = 1).cpu().numpy()      # 'self' is the model itself. We are basically computing the forward pass 'model(x)'

                # Call 'probs_2_label()' to extract the most probable label for each batch element.
                # 'pred_labels' is a numpy array with the most probable labels for every batch element.
                pred_labels.append(probs_2_label(ps))
            #*
            #* END FOR LOOP
            #*##############

        return np.transpose(np.concatenate(pred_labels, axis = 1))


    #*
    #*#### END DEFINED FourLayerNet METHODS #####
    #*###########################################   

#*
#*#### FourLayerNet class  #####
#*##############################

#*###############################
#*#### Conv2DNet class  #####
#*
class Conv2DNet(nn.Module):
    """
    This class define all the layers of the Conv2DNet network and packs them by
    enheriting from the 'Module' class defined by Pytorch.
    - Important: Using 'self' within this class means using the model itself (self = self.model)
    - Important: This network works with 2D data!

    Layers
    ----------

    """

    #*#######################################
    #*#### DEFINED Conv2DNet METHODS #####
    #*
    def __init__(self, num_classes, in_channels):
        """
        Constructor to define the Neural Network model elements. All classes have a function __init__(), which is always executed when the class
	    is bein initiated. It is used to assign vlaues to objet properties or other operations
	    that are necessary to do when the object is being created. 
        
        Inputs
        ----------
        - 'num_classes':    (int) Number of unique labels to classify in the training data.
        - 'in_channels':    (int) Number of channels in the input image.

        , out_channels, kernel_size, stride = 1, padding = 0)

        - 'out_channels':   (int) Number of channels produced by the convolution.
        - 'kernel_size':    (int or tuple) Size of the convolving kernel.
        - 'stride':         (int or tuple) Stride of the convolution.
        - 'padding':        (int or tuple) Zero-padding added to both sides of the input.
        """

        super(Conv2DNet, self).__init__()

        # todo: Properly define the CNN architecture

        #* DEFINE NETWORK LAYERS
        self.conv = nn.Sequential(
            ConvBlock_2D(in_channels=in_channels, out_channels=64),
            ConvBlock_2D(in_channels=64, out_channels=128),
            ConvBlock_2D(in_channels=128, out_channels=256),
            ConvBlock_2D(in_channels=256, out_channels=512),
        )

        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Encapsulate the forward computational steps. PyTorch call the nested model to perform the forward pass.
        DO NOT call the forward(x) method. You should call the whole model itself, as in model(x) to perform a forward pass and output predictions.
        (Within this class, a simple call to self(x) will do the work). It receives a PyTorch tensor as input to compute the pass and then return and output tensor.
        
        Inputs
        ----------
        - x:    Input PyTorch tensor used to compute the forward pass
        """
        x = self.conv(x)
        #x = self.fc(x)
        return x

    def trainNet(self, batch_x, batch_y, epochs = 500, plot = False, lr = 0.002):
        """
        Train the Conv2DNet Neural Network
        
        Inputs
        ----------
        - batch_x:  Python list containing PyTorch tensor batches destined for training
        - batch_y:  Python list containing PyTorch tensor batches labels destined for training
        - epochs:   Number of epochs to run over the training data
        - plot:     Flag to whether or not plot the loss and accuracy per epoch
        - lr:       Learning rate used in the optimizer
        """
    	# Define two empty arrays that will store, for each epoch, the cost and the accuracy
    	# These arrays basically are as big as the number of epochs (or iterations) over the
    	# the training set, meaning that each position on these arrays will have the cost and
    	# the accuracy for every epoch.
    	# They will be used to plot graphs and determine if the training is good or not.
        loss_train = np.zeros(epochs)
        accuracy = np.zeros(epochs)

        # Loss and Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)    # 'self' is the model itself. We are basically doing 'model.parameters()'
        criterion = torch.nn.CrossEntropyLoss()

        # ? GPU FUNCTIONALITY HERE
        # Store the model inside the GPU memory
        #self.cuda()

        # Set the model to train mode to let know PyTorch that during backpropagation
        # it should not apply drop-out, batch norm or any layer with special behaviours
        # that behabe differently on the train and test procedures. 
        self.train()        # 'self' is the model itself. We are basically doing 'model.train()'

        #*#######################################################################
        #* FOR LOOP TO TRAIN THE MODEL WITHT THE CORRESPONDING NUMBER OF EPOCHS
        #* IT ALSO SHOWS A PROGRESS BAR THAT INCREASES ON EVERY EPOCH
        #*
        print("\nStarted training your Neural Network of type: ", str(type(self)))

        for epoch in tqdm(range(epochs)):

            running_loss = 0.0
            correct_train = 0.0

            #*###############################################################
            #* FOR LOOP TO ITERATE OVER ALL TRAINING BATCHES ON EVERY EPOCH
            #*
            for X, Y in zip(batch_x, batch_y):

                """
                print('\n### trainNet() ###')
                print('\t type(X) = ', type(X))
                print('\t X.shape = ', X.shape)
                """

                # ? GPU FUNCTIONALITY HERE
                # Transfer the current batch tensors to the GPU if available
                # X = X.to(device)
                # Y = Y.to(device)

                # Forward pass. This will automatically call the 'forward(self, x)' method
                y_pred = self(X)    # 'self' is the model itself. We are basically doing 'model(X)'

                # ? GPU FUNCTIONALITY HERE
                # Transfer the created tensor to the GPU
                # y_pred = y_pred.to(device)

                # Compute loss.
                # Loss function needs the predicted outputs from 'model()' and a row vector with the same number of elements as the number of entries (rows) in 'y_pred'
                # The problem is that loading 'label4Classes' and convert them as Pytorch tensor, we need to manipulate it, since the shape of the number of labels stored
                # in 'y' has shape of 'torch.Size([10, 1])' when it should be 'torch.Size([10])'. To achieve that, we can transpose the vector to have 1 row and N columns (using '.T') and
                # then extract the first row ('[0]').
                # Please note: 'torch.nn.CrossEntropyLoss()' needs labels starting from 0 to Number of classes -1. That is why we apply -1 since labels should start at 0 and not 1, as saved in 'label4Classes'
                loss = criterion(y_pred, Y.T[0]-1 )

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # Apply softmax and get the value of the tensor with highest probability. Store the predicted class in 'predicted
                predicted =  torch.argmax(F.softmax(y_pred, dim = 1), dim = 1)
                # Sum the correct train of the current mini-batch
                correct_train += ((predicted == Y.T[0]-1).sum().item()) / (predicted.shape[0])
                # Sum the loss of the current mini-batch
                running_loss += loss.item()

                loss_train[epoch] = running_loss/len(batch_x)
                accuracy[epoch] = correct_train/len(batch_x)
            #*
            #* END FOR LOOP
            #*##############
        
        print("Finished training! Your model is now ready to predict.\n")
        #*
        #* END FOR LOOP
        #*##############

        # Plot training loss error
        if(plot):
            plt.title('Train loss and accuracy')
            plt.xlabel('epoch')
            plt.plot(loss_train, 'r-', label = 'loss')
            plt.plot(accuracy, 'g-', label = 'accuracy')
            plt.legend()
            plt.show()

    def predict(self, batch_x):
        """
        Predict 2D batches of data with a Conv2DNet model
        
        Inputs
        ----------
        - 'batch_x':        Python list containing PyTorch tensor batches destined for training
        
        Outputs
        ----------
        - 'pred_labels':    Python list containing numpy arrays with the labels for every element in every batch
        """ 
        #*################
        #* ERROR CHECKER
        #*
        # Check if input parameter is a python list
        if not ( isinstance(batch_x, list) ):
            raise RuntimeError("Expected a python list as input. Received instead element of type: ", str(type(batch_x)) )
        # Check if python list is empty
        if (len(batch_x) == 0):
            raise RuntimeError("Not expected an empty python list input. 'batch_x' is empty.")
        # Check if first python list element is a PyTorch tensor
        if not ( isinstance(batch_x[0], torch.Tensor) ):
            raise TypeError("Expected first element of 'batch_x' to be 'torch.Tensor'. Received instead element of type: ", str(type(batch_x[0])) )
        #*    
        #* END OF ERROR CHECKER ###
        #*#########################

        # Create empty Python lists to store all labels predicted for every batch
        pred_labels = []

        with torch.no_grad():
            self.eval()             # 'self' is the model itself. We are basically doing 'model.eval()' 

            #*##################################################
            #* FOR LOOP TO ITERATE OVER ALL BATCHES
            #* Used to predict the labels for the input batches
            #*
            for X in batch_x:

                # ? GPU FUNCTIONALITY HERE
                # Transfer the current batch tensor to the GPU if available
                # X = X.to(device)

                # For every single batch 'X', we calculate the label probabilities for every element.
                # 'ps' is an array where each row represents the probabilities of each element to be one of the output classes returned by the model.
                # If the batch contains N elements, then 'ps' is an array with N rows and M columns, where M indicates the number of outputs that returns the model.
                ps = F.softmax(self(X), dim = 1).cpu().numpy()      # 'self' is the model itself. We are basically computing the forward pass 'model(x)'

                # Call 'probs_2_label()' to extract the most probable label for each batch element.
                # 'pred_labels' is a numpy array with the most probable labels for every batch element.
                pred_labels.append(probs_2_label(ps))
            #*
            #* END FOR LOOP
            #*##############

        return np.transpose(np.concatenate(pred_labels, axis = 1))

    #*
    #*#### END DEFINED FourLayerNet METHODS #####
    #*###########################################   

#*
#*#### Conv2DNet class  #####
#*##############################

#*##############################
#*#### ConvBlock_2D class  #####
#*
class ConvBlock_2D(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)

        return x
#*
#*#### ConvBlock_2D class  #####
#*##############################


#*#########################
#*#### EXTRA METHODS  #####
#*
def probs_2_label(one_hot_vects):
    """
    Obtain the labels from all passed one hot vectors. Predicted label would be the index + 1 of
    the one hot vector with highest probability

    Inputs
    ----------
    - 'one_hot_vect':   Python list containing PyTorch tensor batches destined for training
    
    Outputs
    ----------
    - Numpy array with labels for every one hot vector
    """ 
    return np.array([np.where(r == np.amax(r))[0] + 1 for r in one_hot_vects]).transpose()

def double_cross_validation(model, k_folds):
    # todo: STRUCTURE TO PERFORM DOUBLE_CROSS_VALIDATION()
    Kn = 0

    best_Kn_OACC = 0
    best_Kn_model = None

    best_K_OACC = 0
    best_K_model = None

    for K in range(0, k_folds, 1):
        print('\n\t\t Current K fold =', K+1)

        for _ in range(0, k_folds, 1):
            # Train CNN in current Kn fold using the calibration data
            # model.trainNet(batch_x = calibration_data_folds[Kn], batch_y = calibration_label_folds[Kn], epochs = epochs, plot = False, lr = 0.01)

            # Test CNN in current Kn fold using the validation data
            # y_hat_Kn = model.predict(batch_x = validation_data_folds[Kn])

            # Evaluate metrics by comparing the predicted labels with the true labels.
            # Use the last column of labels since is the one containing the labels (others has coordenates)
            # Kn_OACC = mts.get_metrics(validation_label_folds[Kn][-1], y_hat_Kn, cm_train.numUniqueLabels)['OACC']
            # if (best_Kn_OACC < Kn_OACC):
                # best_Kn_OACC = Kn_OACC

                # Save Kn CNN model
                # best_Kn_model = # todo: Method to save CNN model in variable!
            

            print('\t\t\t Current Kn fold =', Kn+1)
            Kn += 1

        # Test 'best_Kn_model' with current K test batch
        # y_hat_K = best_Kn_model.predict(batch_x = test_data_folds[K])

        # Evaluate metrics by comparing the predicted labels with the true labels.
        # Use the last column of labels since is the one containing the labels (others has coordenates)
        # K_OACC = mts.get_metrics(test_label_folds[K][-1], y_hat_K, cm_train.numUniqueLabels)['OACC']
        # if (best_K_OACC < K_OACC):
            # best_K_OACC = K_OACC

            # Save K CNN model
            # best_K_model = best_Kn_model

    #return best_K_model
#*
#*#### END EXTRA METHODS  #####
#*#############################