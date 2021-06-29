import json
from json import JSONEncoder
import io

import joblib
import torch
import numpy as np
from PIL import Image

from azureml.core.model import Model

import hsi_dataManager as hsi_dm    # Import 'hsi_dataManager.py' file as 'hsi_dm' to load use all desired functions 
import metrics as mts               # Import 'metrics.py' file as 'mts' to evluate metrics

#*########################
#* AZURE SERVICE ACTIONS
#*

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('Conv2DNet_ID0033C02')
    model = joblib.load(model_path)

# Called when a request is received
def run(json_object):

    # Deserialization
    dictionary = json.loads(json_object)

    raw_image = np.asarray(dictionary['raw_image'])
    white_ref = np.asarray(dictionary['white_ref'])
    black_ref = np.asarray(dictionary['black_ref'])
    patch_size = dictionary['patch_size']
    batch_size = dictionary['batch_size']
    patient_id = dictionary['patient_id']

    # Create an instance of 'RawManager'
    rawManager = hsi_dm.RawManager(raw_image, white_ref, black_ref, patch_size = patch_size, batch_size = batch_size)

    # Preprocess input image
    rawManager.preProcessImage()

    # Extract dimension of the loaded preProcessed cube with added padding for the input image
    dims = rawManager.pad_processedCube.shape

    # Generate batches for feeding the CNN model
    cube_batch = rawManager.create_cube_batch()

    # Convert 'cube' batches to PyTorch tensors for training our Neural Network
    cube_tensor_batch = rawManager.batch_to_tensor(cube_batch['data'], data_type = torch.float)

    # Obtain 'cube' batches coordenates
    cube_coordenates = rawManager.concatenate_list_to_numpy(cube_batch['coords']).astype(int)

    # Predict with the hosted model in the Webservice
    pred_labels = model.predict(batch_x = cube_tensor_batch)

    # Generate classification map from the predicted labels
    title = "Patient " + patient_id + " classification Map"
    fig_predCube, _ = mts.get_classification_map(pred_labels=pred_labels, true_labels=None, coordenates=cube_coordenates, dims=dims, title=title, plot = False, save_plot = False, save_path = None, plot_gt = False, padding=rawManager.pad_margin)

    # Convert a Matplotlib figure to a PIL Image, then cast to Numpy array
    classification_map = fig2numpy(fig_predCube)

    # Return serialized classification map PIL image to bytearray using hexadecimal encoding
    return json.dumps({'classification_map': classification_map}, cls=NumpyArrayEncoder)
    

#*
#* END AZURE SERVICE ACTIONS
#*############################

#*##################
#* fig2numpy method
#*
def fig2numpy(fig):
    """Convert a Matplotlib figure to a numpy array"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi=120)
    buf.seek(0)
    img = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    buf.close()
    return img

#*
#* fig2numpy method
#*###################

#*##########################
#* NumpyArrayEncoder class
#*
class NumpyArrayEncoder(JSONEncoder):
    """
    Class to serialize Numpy arrays to JSON objects.
    Other object instances are enconded by default.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#*
#* END NumpyArrayEncoder class
#*#############################