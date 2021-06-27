import azureml.core
# Packages to load Workspace and get registered Models
from azureml.core import Workspace, Model
# Package to load define Environment dependencies
from azureml.core.conda_dependencies import CondaDependencies 

# Packages for deployment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

import os, shutil

#*###########################
#* CONNECT TO THE WORKSPACE
#*
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


#*###########################
#* GET THE DESIRE MODEL FROM
#* THE WORKSPACE
#*
# Print all models in the workspace 
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')

# Get the model to deploy
model = ws.models['Conv2DNet_test']
print('Loaded ', model.name, 'model version', model.version)


#*###########################
#* CREATE WEBSERVICE TO HOST
#* THE MODEL
#*
folder_name = 'brain_classification_service'

# Create a folder for the web service files
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)
print(folder_name, 'folder created.')

# Copy the necessary Python files into the experiment folder
shutil.copy('./score_brain.py', os.path.join(experiment_folder, "score_brain.py"))
shutil.copy('./preProcessing_chain.py', os.path.join(experiment_folder, "preProcessing_chain.py"))
shutil.copy('./metrics.py', os.path.join(experiment_folder, "metrics.py"))
shutil.copy('./nn_models.py', os.path.join(experiment_folder, "nn_models.py"))
shutil.copy('./hsi_dataManager.py', os.path.join(experiment_folder, "hsi_dataManager.py"))

# Set path for scoring script
script_file = os.path.join(experiment_folder, "score_brain.py")

#*###########################
#* INDICATE CONTAINER HOST
#* TO INSTALL OUR REQUIRED
#* PYTHON DEPENDENCIES
#*

# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib','numpy', 'pillow', 'pip'],
                                                pip_packages=['azureml-sdk', 'azureml-defaults', 'pyarrow', 'torch', 'scipy', 'tqdm'])

# Save the environment config as a .yml file
env_file = os.path.join(experiment_folder,"diabetes_env.yml")
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
# with open(env_file,"r") as f:
#     print(f.read())


#*###########################
#* DEPLOY THE MODEL AS A 
#* WEBSERVICE
#*

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script='score_brain.py',
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = 'brain-service'

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)